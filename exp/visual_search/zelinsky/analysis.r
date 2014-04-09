# Zelinsky search task - data analysis
# Usage: source("analysis.r")

require(ggplot2)

# Parameters
exp_data = "data/0_2014-04-09_05-38-14.csv"  # example (TODO: accept command-line arg)

# Read data
dat = read.csv(exp_data, header=TRUE)
dat = dat[, (colnames(dat) != 'X')]  # drop extra NA col that PsychoPy adds, if any

# Convert appropriate columns to factors
dat$present = factor(dat$present, levels=c(1, 0), labels=c('Positive', 'Negative'))

# Boxplot
rtPlot = ggplot(dat, aes(present, y=response.rt, fill=present)) +
  geom_boxplot() +
  guides(fill=FALSE) +
  xlab("Target presence") + ylab("Response Time (sec)") +
  ggtitle(sprintf("Zelinsky search task\ntarget: %s; display size: %s; presence: %s, #trials: %d",
            paste(unique(dat$target), collapse=", "),
            paste(unique(dat$num_stimuli), collapse=", "),
            paste(levels(dat$present), collapse=", "),
            nrow(dat)))

print(rtPlot)  # actually show plot (if run from command-line, this saves the plot to Rplots.pdf)