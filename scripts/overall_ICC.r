library("cifti")
library("irr")
library("parallel")

num_cores = 16

subjects <- c("103818", "105923", "111312", "114823", "115320", "122317", "125525", "130518", "135528", "137128", "139839", "143325", "146129", "149337", "149741", "151526", "158035", "172332", "175439", "185442", "187547", "192439", "194140", "195041", "200109", "200614", "250427", "287248", "341834", "433839", "562345", "599671", "601127", "660951", "783462", "859671", "861456", "917255")
raters <- c("1200", "Retest")

points_per_map = 64984
maps_per_subject = 47
points_per_subject = points_per_map * maps_per_subject
all_data <- array(double(), c(length(subjects) * points_per_subject, length(raters)))
for (i_subject in 1:length(subjects)){
    subject <- subjects[i_subject]
    start = 1 + (i_subject - 1) * points_per_subject
    end = start + points_per_subject - 1
    for (i_rater in 1:length(raters)){
        rater <- raters[i_rater]
        filename <- paste("/work/aizenberg/dgellis/HCP/HCP_", rater, "/", subject, "/",
                          "T1w/Results/tfMRI_ALL/tfMRI_ALL_hp200_s2_level2.feat/", subject,
                          "_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii", sep="")
        dscalar <- read_cifti(filename)
        all_data[start:end, i_rater] <- dscalar$data
    }
}


# Omit rows that have zero values (due to filtering in preprocessing)
row_sub = apply(all_data, 1, function(row) all(row !=0 ))
# compute ICC results for all the data points
results <- icc(
                             all_data[row_sub,],
                             model = "twoway",
                             type = "consistency",
                             unit = "single"
                             )
print(results)
a = rbind(c("Model", results$model),
      c("Type", results$type),
      c("Subjects", results$subjects),
      c("Raters", results$raters),
      c("ICC", results$value),
      c("95 CI Lower", results$lbound),
      c("95 CI Upper", results$ubound)
     )
write.csv(a, "/work/aizenberg/dgellis/HCP/test-retest_ICC_scores_overall.csv")