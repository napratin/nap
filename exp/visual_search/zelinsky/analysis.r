# Zelinsky search task - data analysis
# Usage: source("analysis.r")

require(ggplot2)

# Parameters (TODO: accept command-line arg)
#exp_data = "data/0_2014-04-09_12-49-03.csv"  # Q, 5
exp_data = "data/0_2014-04-09_06-32-23.csv"  # O, 17
plot_size = c(4, 4)

# Read data
dat = read.csv(exp_data, header=TRUE)
dat = dat[, (colnames(dat) != 'X')]  # drop extra NA col that PsychoPy adds, if any

# Convert appropriate columns to factors
dat$present = factor(dat$present, levels=c(1, 0), labels=c('Positive', 'Negative'))

# Data cleaning
dat = dat[dat$response.corr==1,]

# Boxplot
rtPlot = ggplot(dat, aes(present, y=response.rt, fill=present)) +
  geom_boxplot() +
  #coord_cartesian(ylim=quantile(dat$response.rt, c(0.0001, 0.99))) +  # prevent outliers from skewing plot
  guides(fill=FALSE) +
  xlab("Target presence") + ylab("Reaction Time (sec)") +
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
