# Zelinsky search task - data analysis
# Usage: source("analysis.r")

require(ggplot2)

# Parameters (TODO: accept command-line arg)
drop_incorrect = TRUE  # drop trials with incorrect answer
plot_size = c(7, 4)  # c(4, 4) for single plots, c(7, 4) for side-by-side

## Input data: One combined file
data_file = "data/all_2014-04-09.csv"  # uncomment to read this single file

## Input data: Individual runs to combine
Q_5 = "data/Q_5_256_0_2014-04-09_12-49-03.csv"
Q_17 = "data/Q_17_256_0_2014-04-09_16-05-42.csv"
O_5 = "data/O_5_256_0_2014-04-09_17-56-49.csv"
O_17 = "data/O_17_256_0_2014-04-09_18-58-21.csv"
#data_file_list = c(Q_5, Q_17, O_5, O_17)  # uncomment to read all files in this list

# Utility functions
## Standard error of the mean (stdErr); for error bars: lower limit (errBarLow), upper limit (errBarHigh)
stdErr <- function(x) { sqrt(var(x, na.rm=TRUE) / length(na.omit(x))) }
errBarLow <- function(x) { return(mean(x) - stdErr(x)) }
errBarHigh <- function(x) { return(mean(x) + stdErr(x)) }

# TODO: Compute within subject variablility error bars as per Zelinksy paper

## Data reading
readData <- function(filename) {
	cat("Reading data file:", filename, "\n")
	dat_ = read.csv(filename, header=TRUE)
	dat_ = dat_[, (colnames(dat_) != 'X')]  # drop extra NA col that PsychoPy adds, if any
	return(dat_)
}

readDataAll <- function() {
	dat_all = NULL
	for (filename in data_file_list) {
		dat_ = readData(filename)
		if(is.null(dat_all)) {
			dat_all = dat_
		}
		else {
			dat = rbind(dat_all, dat_)
		}
	}
	return(dat_all)
}

# Read data
if(exists("data_file")) {
	dat = readData(data_file)
} else if(exists("data_file_list")) {
	dat = readDataAll(data_file_list)
}

# Convert appropriate columns to factors
if(!is.factor(dat$present)) {
	dat$present = factor(dat$present, levels=c(1, 0), labels=c('Positive', 'Negative'))  # convert
} else if(all(levels(dat$present) == c('Negative', 'Positive'))) {
	dat$present <- factor(dat$present, levels=c('Positive', 'Negative'), labels=c('Positive', 'Negative'))  # re-order
}

# Data cleaning
if (drop_incorrect) {
	dat = dat[dat$response.corr==1,]
}

# Data summary
#dat_sum <- summarySE(dat, measurevar="response.rt", groupvars=c("target", "num_stimuli"))

# Plot
rtPlot = ggplot(dat, aes(x=as.factor(num_stimuli), y=response.rt, colour=target, shape=target)) +
  stat_summary(aes(group=target), fun.y=mean, geom="line", size=0.75) +
  stat_summary(fun.y=mean, geom="point", size=1.25) +
  stat_summary(fun.ymin=errBarLow, fun.ymax=errBarHigh, geom="errorbar", width=0.125) +
  facet_grid(. ~ present) +
  scale_colour_discrete(name="Task", breaks=c('Q', 'O'), labels=c("Parallel", "Serial")) +
  scale_shape_discrete(name="Task", breaks=c('Q', 'O'), labels=c("Parallel", "Serial")) +
  guides(colour=guide_legend(title=NULL), shape=guide_legend(title=NULL)) +
  xlab("Display size") + ylab("Reaction Time (sec)") +
  ggtitle(sprintf("Zelinsky search task\ntarget: %s; display size: %s; #trials: %d",
            paste(unique(dat$target), collapse=", "),
            paste(unique(dat$num_stimuli), collapse=", "),
            nrow(dat))) + 
  # To include target presence levels: paste(levels(dat$present), collapse=", ")
  theme(title=element_text(size=10),
  		axis.text=element_text(size=10),
        axis.title=element_text(size=10))

dev.new(width=plot_size[1], height=plot_size[2])
#X11()  # maybe useful for launching plots from command-line
print(rtPlot)  # actually show plot (if run from command-line, this saves the plot to Rplots.pdf)
