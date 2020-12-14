# Detecting-asymmetric-information-with-dbg-pds
Building on the dbg-pds-option-calibration git, I am now using the data in order to detect cases of asymmetric information

# The BuildInputs class is just here to transform the parameters files generated in the dbg-pds-option-calibration git into time series. As the parameters were generated each time a batch of 5 trades could be used to calibrate a set of parameters for any given maturity, the szerie was unevenly sampled. Thanks to some extrapolation (1 day max) we can now have a full time series of parameters sampled on 1 minute intervals.
Separately, we also compute time series deriving from the actual trades (the sames that we used to calibrate the vols parameters). The tyraded instruments are prices using the parameters and the 
This series is then merged in the same dataframe as Pricing class in order to collect the associated sensitivity (vega, sega, div sensitivity). The idea here is to later associate a jump in traded volume on one sensitivity to a later surge in the associated parameter if someone had the information leading to this surge before the rest of market participants.
Going further still, these sensitivities are "signed" in order to (try to) signify if it was a buying or selling interest. The disputable hypothesis here is that if the parameter is going up, then it means that trades happening at this time are responsible for this trend (since trades are what is used to calibrate) so these must be buying the sensitivity (vega or other). 
Once merged into the df_pivot dataframe, those inputs will be used to compute the actual X and Y for our machine learning project.

# The BuildXY class aims to convert those time series into "stationary" ones.
The idea here is that we are trying to detect a trading pattern that may be common to all instances of asymmetric information. We need to iron out everything that is specific to a certain period or stock along with false positive in order to get clean data. Here are the steps taken

1/ Look at forward parameters :
This cleans out any punctual event like a dividend ex date that will inevitably prompt a jump in the spot/forward ratio but not in a  forward - spot/forward ratio.
Similarly, if a company publishes its earnings report, the vol will collapse but hte forward vol will be mostly unaffected, preventing false positives.This is what the differentiate_matu function does.
2/ Short term trend divided by long term trend:This simply uses exponentially weighted moving average in order to detect short term variation of volume/level from a long term average.This is done by the  differentiate_time functionThis is also where the Y series is generated : it simply consist in 
3/ divide by the corresponding time series of the euro stoxx 50 index in order to iron out market regimes.

# Finally, a very simple Dense Neural Network with Keras will show that those inputs are by no means sufficient to detect a pattern that would annonce a later sudden shift of parameters.Cases of sudden move of a parameter (heere 3 standard deviations) are rare and oonsome of those events would have been preceded by abnormal trading patterns by well informed agents so there is  enough data to feed a complex neural network for fear of over-parameterization.
