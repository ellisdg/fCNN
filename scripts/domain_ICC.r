library("cifti")
library("irr")
library("parallel")
library("hash")

num_cores = 16

subjects <- c("103818", "105923", "111312", "114823", "115320", "122317", "125525", "130518", "135528", "137128", "139839", "143325", "146129", "149337", "149741", "151526", "158035", "172332", "175439", "185442", "187547", "192439", "194140", "195041", "200109", "200614", "250427", "287248", "341834", "433839", "562345", "599671", "601127", "660951", "783462", "859671", "861456", "917255")
raters <- c("1200", "Retest")

points_per_map = 64984
maps_per_subject = 47
points_per_subject = points_per_map * maps_per_subject
all_data_3 <- array(double(), c(length(subjects) * points_per_map, length(raters), maps_per_subject))
# for (i_subject in 1:length(subjects)){
for (i_subject in 1:2){
    subject <- subjects[i_subject]
    start = 1 + (i_subject - 1) * points_per_map
    end = start + points_per_map - 1
    for (i_rater in 1:length(raters)){
        rater <- raters[i_rater]
        filename <- paste("/work/aizenberg/dgellis/HCP/HCP_", rater, "/", subject, "/",
                          "T1w/Results/tfMRI_ALL/tfMRI_ALL_hp200_s2_level2.feat/", subject,
                          "_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii", sep="")
        dscalar <- read_cifti(filename)
        for (i_map in 1:maps_per_subject){
            all_data_3[start:end, i_rater, i_map] <- dscalar$data[ ,i_map]
            }
    }
}

domains <- hash()
domains["Motor"] <- 1:13
domains["Language"] <- 14:16
domains["WM"] <- 17:35
domains["Relational"] <- 36:38
domains["Emotion"] <- 39:41
domains["Social"] <- 42:44
domains["Gambling"] <- 45:47

df <- data.frame(domain=character(),
                 ICC=double(),
                 lbound=double(),
                 ubound=double(),
                 subjects=integer(),
                 raters=integer(),
                 stringsAsFactors=FALSE)
for (key in keys(domains)){
    data <- array(double(), c(0, length(raters)))
    for (i in values(domains, keys=key)[,]){
        data <- rbind(data, all_data_3[ , , i])
    }
    # Omit rows that have zero values (due to filtering in preprocessing)
    row_sub = apply(data, 1, function(row) all(row !=0 ))
    # compute ICC results for all the data points
    results <- icc(data[row_sub,],
                   model = "twoway",
                   type = "consistency",
                   unit = "single")

    df[nrow(df) + 1, ] <- c(key, results$value, results$lbound, results$ubound, results$subjects, results$raters)
}
write.csv(df, "/work/aizenberg/dgellis/HCP/test-retest_ICC_scores_domain.csv")