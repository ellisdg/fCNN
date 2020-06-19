library("cifti")
library("irr")
library("parallel")

num_cores = 16

subjects <- c("103818", "105923", "111312", "114823", "115320", "122317", "125525", "130518", "135528", "137128", "139839", "143325", "146129", "149337", "149741", "151526", "158035", "172332", "175439", "185442", "187547", "192439", "194140", "195041", "200109", "200614", "250427", "287248", "341834", "433839", "562345", "599671", "601127", "660951", "662551", "783462", "859671", "861456", "917255")
raters <- c("1200", "Retest")

all_data <- array(double(), c(length(subjects), length(raters), 64984, 47))


for (i_subject in 1:length(subjects)){
    subject <- subjects[i_subject]
    for (i_rater in 1:length(raters)){
        rater <- raters[i_rater]
        filename <- paste("/work/aizenberg/dgellis/HCP/HCP_", rater, "/", subject, "/",
                          "T1w/Results/tfMRI_ALL/tfMRI_ALL_hp200_s2_level2.feat/", subject,
                          "_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii", sep="")
        dscalar <- read_cifti(filename)
        all_data[i_subject, i_rater, , ] <- dscalar$data
    }
}

map_names <- dscalar$NamedMap$map_names
n_points <- dim(all_data)[3]

icc_scores <- array(double(), c(length(map_names), n_points))

for (i_map in 1:length(map_names)){
    func <- function(i_point, map_data){
        data <- map_data[ , , i_point]
        if (all(data != 0)){
            icc_score <- icc(
                             data, 
                             model = "twoway", 
                             type = "consistency", 
                             unit = "single"
                             )

            value <- icc_score[7]$value
        }
        else{
            value <- -1
        }
        return(value)
    }
    map_icc_scores <- mclapply(1:n_points, func, mc.cores=num_cores, map_data=all_data[ , , , i_map])
    icc_scores[i_map, ] <- unlist(map_icc_scores)
}

df <- data.frame(icc_scores)

write.csv(df, "/work/aizenberg/dgellis/HCP/test-retest_ICC_scores.csv")
