# Context :

My goal is to use this output to detect instances of asymmetrical information in the stock options market.

Asymmetrical information can stem from criminal behaviours like insider trading but also from an edge given by advanced research to actors deploying extensive means like the use of mobile phone data, private polls or other types of intelligence gathering along with machine learning treatment of those data.
The development of those techniques risks undermining the business model of less specialized actors, including market makers, thus jeopardizing the structure of the market.

Asymmetrical information can be detected in restrospect because they will ultimately lead to a dramatic shift of a parameter such as the spot price, the volatility or the dividend yield.

The goal here is to identify signals in the trading pattern that will alert liquidity providers that something is fishy.

# Detecting-asymmetric-information-with-dbg-pds

Building on the dbg-pds-option-calibration git, I am now using the data in order to detect instances of asymmetric information.


# The BuildInputs class is just here to transform the parameters files generated in the dbg-pds-option-calibration git into time series. 
As the parameters were generated each time a batch of 5 trades could be used to calibrate a set of parameters for any given maturity.
This leads to a serie that is unevenly sampled. Thanks to some extrapolation (1 day max) the BuildInputs class transform these data into a proper time series of parameters sampled on 1 minute intervals.
Separately, we also compute time series deriving from the actual trades (the sames that we used to calibrate the vols parameters).

This series is then merged in the same dataframe as Pricing class in order to collect the associated sensitivity (vega, sega, div sensitivity). The idea here is to later associate a jump in traded volume on one parameter to a later surge in the associated parameter : This will happen if someone had the information leading to this surge before the rest of market participants and tried to take advantage of this knowledge.

Going further still, these sensitivities are "signed" in order to (try to) signify if it was a buying or selling interest. The disputable hypothesis here is that if the parameter is going up, then it means that trades happening at this time are responsible for this trend (since trades are what is used to do the calibration) so these trades must be buying the sensitivity (vega or other). 
Once merged into the df_pivot dataframe, those inputs will be used to compute the actual X and Y for our machine learning project.

# The BuildXY class aims to convert those time series into "stationary" ones.
The idea here is that we are trying to detect a trading pattern that may be common to all instances of asymmetric information. We need to iron out everything that is specific to a certain period or stock, along with false positives, in order to get clean data. Here are the steps taken :

1/ Look at forward parameters :
This cleans out any punctual event like a dividend ex date that will inevitably prompt a jump in the spot/forward ratio but not in a  "forward - spot/forward ratio" which is computed as the ratio of two spot/forward ratio on different maturities (typically a long term one divided by the nearby).
Similarly, when a company publishes its earnings report, the implicit vol will collapse but the forward vol will be mostly unaffected on average, which is what we want. This is what the differentiate_matu function does.

2/ Short term trend divided by long term trend :
This simply uses exponentially weighted moving average in order to detect short term variation of volume/level from a long term average. This is done by the  differentiate_time function. This is also where the Y series is generated : it simply consist in observing parameters variation over the 5 days following the date at which we oberve the Xs.

3/ divide by the corresponding time series of the euro stoxx 50 index :
This will iron out market regimes.

# Finally, a very simple Dense Neural Network with Keras will show that those inputs are by no means sufficient to detect a pattern that would annonce a later sudden shift of parameters.
Cases of sudden move of a parameter (heere 3 standard deviations) are rare and only some of those events would have been preceded by abnormal trading patterns by well informed agents so there is  enough data to feed a complex neural network for fear of over-parameterization.
