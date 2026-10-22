# AI Analyst Authoritative Project Knowledge

This document is the primary explanatory knowledge source for the Taxi-Out Prediction AI Analyst. Numerical answers should use the live dataset or the structured project files when available. The values stated here reflect the current project artifacts.

## Project identity and objective

The project predicts aircraft taxi-out duration and analyzes airport surface congestion. Taxi-out duration is the elapsed time in minutes between aircraft pushback from the gate and wheels-off at takeoff.

The target variable is `TAXI_OUT`:

`TAXI_OUT = WHEELS_OFF - DEP_TIME`

The prediction supports airport collaborative decision-making, departure sequencing, gate-hold evaluation, estimated wheels-off time, and analysis of surface queues. It is an academic decision-support prototype and does not replace certified airline-dispatch or air-traffic-control systems.

## Dataset scope and source

The project uses 2025 Bureau of Transportation Statistics Marketing Carrier On-Time Performance records. The sample contains major United States airports and is concentrated around Dallas/Fort Worth International Airport and American Airlines. Airport and carrier comparisons should therefore be interpreted together with their flight counts.

The raw dataset contains 75,000 flight records. After cleaning, 72,272 records remain, giving a retention rate of 96.36 percent.

The principal raw fields are:

| Field | Meaning |
|---|---|
| `FL_DATE` | Flight date |
| `OP_UNIQUE_CARRIER` | Operating-carrier code |
| `ORIGIN` | Departure-airport code |
| `DEST` | Arrival-airport code |
| `CRS_DEP_TIME` | Scheduled departure time in HHMM format |
| `DEP_TIME` | Actual departure or pushback time in HHMM format |
| `DEP_DELAY` | Departure delay in minutes |
| `TAXI_OUT` | Pushback-to-wheels-off duration in minutes |
| `WHEELS_OFF` | Takeoff time |
| `DISTANCE` | Flight distance in miles |
| `CANCELLED` | Cancellation indicator |
| `DIVERTED` | Diversion indicator |

## Data cleaning and quality controls

Cleaning removes flights that cannot represent normal taxi-out operations:

| Cleaning rule | Records removed | Reason |
|---|---:|---|
| Cancelled flights | 1,507 | The aircraft did not complete normal departure operations |
| Diverted flights | 194 | Irregular operations do not represent the target process |
| Missing, zero, or negative taxi-out values | 438 | The target is absent or physically invalid |
| Taxi-out duration above 120 minutes | 589 | Extreme disruption is outside the modeled operating range |

The outlier strategy filters values above 120 minutes. It does not replace them with 120 minutes.

Before outlier treatment, the taxi-out distribution has a median of 23.4 minutes, a mean of approximately 25.56 minutes, and a maximum of 174.8 minutes. There are 1,476 records above 90 minutes and 589 records above 120 minutes.

After cleaning, taxi-out time has a mean of approximately 24.57 minutes, a median of 23.4 minutes, a standard deviation of approximately 9.88 minutes, a minimum of 11.7 minutes, and a maximum of 119.0 minutes.

## Prediction timing and leakage control

The prediction is intended for use before or at gate pushback. The system uses information available at that stage, including schedule, pushback delay, route distance, recent origin congestion, carrier, airport, and temporal context.

The data split is chronological rather than random:

| Partition | Period | Records | Share |
|---|---|---:|---:|
| Training | January through October 2025 | 60,271 | 83.4 percent |
| Test | November through December 2025 | 12,001 | 16.6 percent |

This out-of-time split evaluates the models on later flights and reduces future-information leakage. Target encodings are calculated only from the training partition and then mapped to the test partition.

## Model input features

The final model feature set contains eight continuous features and three categorical features.

### Continuous features

| Feature | Meaning |
|---|---|
| `DEP_DELAY` | Departure delay already present before or at pushback |
| `DISTANCE` | Scheduled flight distance in miles |
| `CONGESTION_1H` | Origin-airport pushbacks during the preceding 60 minutes |
| `CONGESTION_30M` | Origin-airport pushbacks during the preceding 30 minutes |
| `SIN_HOUR` | Sine encoding of scheduled departure hour |
| `COS_HOUR` | Cosine encoding of scheduled departure hour |
| `ORIGIN_TAXI_MEAN` | Smoothed historical taxi-out mean for the origin airport |
| `CARRIER_TAXI_MEAN` | Smoothed historical taxi-out mean for the operating carrier |

### Categorical features

| Feature | Meaning |
|---|---|
| `OP_UNIQUE_CARRIER` | Operating airline code |
| `ORIGIN` | Departure-airport code |
| `DAY_OF_WEEK` | Weekday derived from the flight date, where Monday is 0 and Sunday is 6 |

`DEST` is retained for route analysis and display but is not included in the current final model feature list. `DEP_TIME` supports estimated wheels-off calculation but is not used directly as a model feature.

Continuous variables pass through `StandardScaler`. Categorical variables pass through one-hot encoding with unknown values ignored.

## Cyclical time encoding

Scheduled departure hour is represented by sine and cosine transformations. This prevents midnight from appearing artificially distant from 23:00 and allows the models to learn recurring daily departure-bank patterns.

`SIN_HOUR = sin(2 * pi * hour / 24)`

`COS_HOUR = cos(2 * pi * hour / 24)`

## Congestion features

`CONGESTION_1H` measures the number of origin-airport pushbacks in the previous hour. It represents sustained departure-surface demand.

`CONGESTION_30M` measures pushbacks during the previous 30 minutes. It identifies recent demand surges within the broader hourly level.

The dashboard uses these descriptive congestion categories:

| One-hour pushbacks | Congestion description |
|---:|---|
| Less than 8 | Low or free flow |
| 8 to 14 | Moderate or standard push |
| 15 to 21 | High or peak bank |
| 22 or more | Severe surface queueing |

These congestion categories describe traffic density. They are different from the taxi-out risk bands used by the operational confusion matrix.

## Live congestion sensitivity curve

The Live Congestion Sensitivity Curve is a scenario-based what-if chart. It shows how the selected model's predicted taxi-out time responds when origin-airport pushback congestion changes while the other selected flight inputs remain fixed.

The dashboard calculates the curve by varying `CONGESTION_1H` from 1 through 35 pushbacks. For each value, it sets `CONGESTION_30M` to the integer half of the hourly value, prepares the same selected flight with those congestion inputs, and generates a new model prediction.

The horizontal axis is origin pushbacks during the previous hour. The vertical axis is predicted taxi-out time in minutes. A dashed vertical line identifies the congestion value currently selected in the dashboard.

An upward curve means the model predicts longer taxi-out time as surface traffic increases. A steep section means the prediction is more sensitive to additional pushbacks in that range. A flat section means added congestion changes the prediction only slightly. Tree models can produce steps or small non-monotonic movements because they learn nonlinear rules and feature interactions.

The curve represents model sensitivity for the selected flight scenario. It is not a historical time series, it does not predict how many pushbacks will occur, and it does not establish that congestion alone caused the predicted change.

## Target encoding

Origin and carrier target encodings represent historical taxi-out behavior. The project uses smoothed empirical-Bayes means with smoothing parameter 10. The encoding combines the group mean with the global training mean, reducing instability for groups with few observations.

The encodings are fitted only on January-through-October training records. Unknown origins or carriers use the global mean. If the stored global mean is unavailable during inference, the fallback value is 20.0 minutes.

## Models

The project evaluates three regression models.

### Linear Regression baseline

Linear Regression provides an interpretable reference model. It represents additive linear relationships and establishes whether the ensemble methods improve on a simple baseline.

### Random Forest Regressor

Random Forest combines many decision trees and captures nonlinear relationships and feature interactions. Current parameters are 100 trees, maximum depth 15, minimum samples per split 10, and random state 42.

### XGBoost Regressor

XGBoost builds trees sequentially to reduce the residual errors of earlier trees. Current parameters are 100 estimators, maximum depth 6, learning rate 0.1, and random state 42.

## Model evaluation metrics

The project reports metrics on both training data and the November-December out-of-time test set.

| Metric | Interpretation | Preferred direction |
|---|---|---|
| MAE | Mean absolute prediction error in minutes | Lower is better |
| RMSE | Root mean squared error, which penalizes large errors more strongly | Lower is better |
| R-squared | Share of target variance explained relative to a mean baseline | Higher is better |
| Within 3 minutes | Percentage of predictions within plus or minus 3 minutes | Higher is better |
| Within 5 minutes | Percentage of predictions within plus or minus 5 minutes | Higher is better |
| Mean residual bias | Average signed prediction error | Closer to zero is better |

## Test-set model comparison

| Model | MAE | RMSE | R-squared | Within 3 minutes | Within 5 minutes |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 4.170 min | 10.037 min | 0.0166 | 54.60 percent | 78.91 percent |
| Random Forest | 4.040 min | 9.999 min | 0.0239 | 57.48 percent | 82.60 percent |
| XGBoost | 3.879 min | 9.890 min | 0.0452 | 58.80 percent | 84.88 percent |

XGBoost is the best current test model. It has the lowest test MAE and RMSE, the highest test R-squared, and the highest percentages within 3 and 5 minutes. Random Forest ranks second on the current test results, and Linear Regression is the baseline.

The dashboard may allow the user to select any of the three models. A selectable or recommended dashboard model is not necessarily the best model according to the current out-of-time metrics. Questions about model performance should use the test metrics above.

## Why the R-squared values are low

Taxi-out duration contains operational variation that is absent from public BTS schedule records. Unobserved factors include real-time runway configuration, weather, de-icing, ramp conflicts, tug availability, runway crossings, tactical air-traffic-control holds, and departure-metering changes.

Rare long delays also widen the error distribution. Therefore, R-squared remains low even when most predictions are close to the observed time. MAE and the percentage within 3 or 5 minutes provide direct operational interpretations, but R-squared should still be reported honestly.

## Random Forest feature importance

Random Forest impurity-based feature importance from the current trained model is:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `DISTANCE` | 30.03 percent |
| 2 | `DEP_DELAY` | 18.27 percent |
| 3 | `SIN_HOUR` | 8.81 percent |
| 4 | `CONGESTION_1H` | 6.73 percent |
| 5 | `COS_HOUR` | 6.42 percent |
| 6 | `CONGESTION_30M` | 5.23 percent |
| 7 | `ORIGIN_TAXI_MEAN` | 3.36 percent |
| 8 | `CARRIER_TAXI_MEAN` | 2.38 percent |
| 9 | `DAY_OF_WEEK_2` | 1.99 percent |
| 10 | `DAY_OF_WEEK_4` | 1.97 percent |

Feature importance shows how strongly the fitted Random Forest used each input to split the training data. It does not prove causation and does not show whether a feature increases or decreases taxi-out time.

## XGBoost feature importance

The five highest current XGBoost importances are:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `SIN_HOUR` | 13.98 percent |
| 2 | `COS_HOUR` | 8.62 percent |
| 3 | `ORIGIN_TAXI_MEAN` | 7.11 percent |
| 4 | `CARRIER_TAXI_MEAN` | 6.13 percent |
| 5 | `DAY_OF_WEEK_6` | 5.39 percent |

Random Forest and XGBoost can rank features differently because they build and combine trees differently. Their importance scales should be interpreted within each model.

## Linear Regression coefficients

Linear Regression coefficients describe the fitted linear association after preprocessing. A positive coefficient increases the prediction when the corresponding transformed feature increases, while a negative coefficient decreases it. Coefficients are not directly comparable with tree-model feature importances and do not establish causation.

The largest current coefficient magnitudes include `ORIGIN_DFW`, `ORIGIN_TAXI_MEAN`, `CONGESTION_1H`, `ORIGIN_JFK`, `CARRIER_TAXI_MEAN`, and `DISTANCE`.

## Operational taxi-out risk bands

The dashboard converts continuous actual and predicted taxi-out times into four operational bands:

| Risk band | Taxi-out duration | Interpretation |
|---|---:|---|
| Normal | Less than 20 minutes | Normal surface flow |
| Moderate | 20 to less than 30 minutes | Developing queue |
| High | 30 to less than 45 minutes | Material surface congestion |
| Severe | 45 minutes or more | Major delay or gridlock risk |

These are reporting bands applied to regression results. The underlying models still predict continuous taxi-out minutes.

## Confusion matrix

The operational confusion matrix compares actual risk bands with predicted risk bands on the test predictions. Rows represent actual bands and columns represent predicted bands.

Diagonal cells count flights assigned to the correct risk band. Off-diagonal cells count flights placed in a different band. Cells next to the diagonal usually represent smaller operational errors than cells far from the diagonal. The matrix complements MAE and RMSE by showing whether prediction errors change the operational congestion category.

The confusion matrix is a classification view of regression output. It does not mean that the regression models were trained as classifiers.

## Prediction inputs and outputs

The dispatch simulator accepts carrier, origin, destination, flight date, scheduled departure, actual pushback, departure delay, distance, one-hour congestion, and 30-minute congestion.

The prediction output contains:

| Output | Meaning |
|---|---|
| Predicted taxi-out | Estimated minutes from pushback to wheels-off |
| Estimated wheels-off | Actual pushback time plus predicted taxi-out duration |
| Scheduled departure | Formatted scheduled time |
| Actual pushback | Formatted departure time |
| Model used | Selected regression model |

The prediction is an estimate, not an air-traffic-control clearance or guaranteed takeoff time.

## Dashboard functions

The Streamlit application provides data-quality information, a dispatch prediction simulator, model benchmarks, an operational risk confusion matrix, feature-importance views, congestion analysis, route and airport summaries, and the AI Analyst chat interface.

Scenario presets include a DFW morning peak, a DFW midday off-peak case, and a congested JFK evening case. Presets populate the inputs and may be edited before prediction.

## Complete dashboard chart and value interpretation guide

The following sections define every analytical chart, KPI card, prediction value, and supporting number displayed by the current dashboard. Unless a section explicitly says that values are configured references, displayed values are calculated from the cleaned data, the selected flight inputs, or the stored model-evaluation artifacts.

### Raw Records

Raw Records is the number of flight rows before cleaning. The current value is 75,000. It is the starting denominator for the cleaning audit and should not be interpreted as the number of flights used for model training.

### Clean Records

Clean Records is the number of usable rows remaining after cancelled flights, diverted flights, invalid taxi-out targets, and taxi-out observations above 120 minutes are removed. The current value is 72,272. These records form the cleaned analytical dataset before the chronological train-test split.

### Retention Rate

Retention Rate is Clean Records divided by Raw Records, multiplied by 100. The current rate is 96.36 percent. A high value indicates that most raw observations passed the quality rules; it does not by itself prove that the retained data is unbiased or complete.

### Mean Taxi-Out

Mean Taxi-Out is the arithmetic average of TAXI_OUT across the cleaned records. The current mean is approximately 24.57 minutes. It describes the overall center of the cleaned sample but can be pulled upward by long taxi-out events, so it should be read with the median and percentiles.

### Data Cleaning Funnel

The Data Cleaning Funnel is a horizontal bar chart showing how many records remain after each data-quality filter. Its stages are Raw records, After cancelled removal, After diverted removal, After invalid target removal, and Final cleaned records. The current sequence is 75,000, 73,493, 73,299, 72,861, and 72,272 records.

The drop between two adjacent bars is the number removed at that step: 1,507 cancelled flights, 194 diverted flights, 438 missing, zero, or negative taxi-out targets, and then 589 taxi-out outliers above 120 minutes before the final count. The outlier step is included in the Final cleaned records bar even though it does not have a separately named intermediate bar.

The chart tells you how much data each rule excludes and whether cleaning materially reduces the sample. It is a data lineage and quality-control view, not a model-accuracy chart. A narrowing funnel is expected; an unexpectedly large drop would indicate a possible data-quality or rule-design issue.

### Taxi-Out Target Reference

The Taxi-Out Target Reference chart summarizes the taxi-out distribution using Minimum, Median, Mean, 75th percentile, 90th percentile, 95th percentile, and Maximum values. The minimum, median, mean, and maximum describe the post-cleaning target, while the displayed P75, P90, and P95 reference the audit distribution inspected before outlier treatment.

The median is the middle flight, the mean is the arithmetic average, and a percentile is the value at or below which that percentage of observations falls. For example, P90 is the taxi-out time at or below which 90 percent of observations lie. The chart shows the center, spread, and upper tail of taxi-out duration; it does not show the number of flights at each value.

### Carrier Mean Taxi-Out

The carrier comparison bar chart displays average cleaned taxi-out time for each operating carrier. A longer bar means that the carrier has a higher sample mean. Differences may reflect airport mix, schedules, route mix, departure banks, and sample size, so the chart describes association rather than a causal carrier effect.

### Mean Taxi-Out by Departure Hour

The departure-hour line chart groups cleaned flights by scheduled departure hour and plots average taxi-out minutes. Peaks can reveal hours associated with departure banks or heavier surface demand. It is an aggregate historical profile; it does not mean that every flight in a peak hour will experience that delay.

### Highest Mean Taxi-Out Origins

This horizontal bar chart ranks origin airports by average taxi-out duration. Hover information includes the number of flights used for each airport. A high mean identifies an airport with longer taxi-out in this sample, but comparisons should be interpreted with flight volume and operational mix.

### Monthly Taxi-Out Trend

The monthly line chart displays mean taxi-out by calendar month. Hover values include Flights, P25, P50 or median, P75, P90, P95, maximum, and minimum. It reveals seasonal movement and changes in distribution, but a monthly change can also reflect a changing airport, carrier, or route mix.

### Predicted Taxi-Out

Predicted Taxi-Out is the selected regression model's estimate of minutes from actual pushback to wheels-off for the current input flight. It is conditional on the entered flight, time, airport, carrier, distance, delay, and congestion information. It is an estimate rather than a guaranteed takeoff time or operational clearance.

### Actual Pushback and Scheduled Departure

Actual Pushback is the entered DEP_TIME, shown as a clock time. The supporting Scheduled Departure value is the entered CRS_DEP_TIME, and the signed minute value in parentheses is DEP_DELAY. A positive delay means pushback occurred after schedule; a negative value means it occurred before schedule.

### Estimated Wheels-Off

Estimated Wheels-Off is calculated by adding Predicted Taxi-Out to Actual Pushback. It is a derived clock time for planning and coordination. It is not an observed timestamp and does not account for new events that occur after the prediction is made.

### Surface Congestion

Surface Congestion shows the entered one-hour pushback count and maps it to a descriptive band: fewer than 8 is Low or Free Flow, 8 to 14 is Moderate or Standard Push, 15 to 21 is High or Peak Bank, and 22 or more is Severe Queueing. This label describes input traffic density and is separate from the taxi-out risk bands used in the confusion matrix.

### Pushback-to-Airborne Sequence Breakdown

The Pushback-to-Airborne Sequence Breakdown compares a heuristic Base or Unimpeded Taxi estimate, an estimated Queue and Congestion Delay contribution, and Total Predicted Taxi-Out. Total Predicted Taxi-Out is the model output. The base value is calculated as the prediction minus 0.45 times CONGESTION_1H and minus 0.1 times positive DEP_DELAY, with a 10-minute minimum; queue contribution is the remaining difference.

This chart is an explanatory dashboard decomposition, not a direct output from separate model components and not a measurement of physical taxi and queue stages. It helps users reason about the displayed prediction, but its component values should not be treated as independently validated causal quantities.

### Live Congestion Sensitivity Curve

The Live Congestion Sensitivity Curve is a scenario-based what-if chart showing how predicted taxi-out changes as one-hour pushbacks vary from 1 to 35 while the selected flight and model remain fixed. The horizontal axis is pushbacks in the preceding hour, the vertical axis is predicted taxi-out minutes, and the dashed vertical line marks the current input.

An upward or steep section indicates greater model sensitivity to added surface demand. A flat section indicates little predicted change. Steps or small reversals can occur with tree models. The curve is not a historical time series and does not prove that congestion alone caused the change.

### Dispatch Decision Support scenario chart

The Dispatch Decision Support bar chart compares four model scenarios: Low at 5, Moderate at 15, Severe at 25, and Extreme at 35 one-hour pushbacks. The same selected flight is rescored at each scenario, with 30-minute congestion set to half of the hourly value.

The bars show model-based what-if predictions rather than averages of observed flights. Their purpose is to show whether the selected flight's estimated taxi-out materially changes across congestion conditions.

### Current prediction, Queue contribution, Impact versus low congestion, and Operational status

Current prediction repeats the selected flight's predicted taxi-out. Queue contribution is the heuristic difference between the total prediction and the calculated base or unimpeded value. Impact versus low congestion is the current prediction minus the scenario prediction at 5 pushbacks per hour; a positive value means the current congestion setting produces a longer prediction than the low scenario.

Operational status repeats the congestion-density label derived from the entered one-hour pushbacks. The recommendation below these values follows the same thresholds: normal planning below 8, standard coordination from 8 to 14, close coordination from 15 to 21, and possible departure metering or a short pushback hold at 22 or more.

### Comparative Model Scores

The Comparative Model Scores table reports November-December out-of-time test results for Linear Regression, Random Forest, and XGBoost. MAE and RMSE are errors in minutes and lower values are better. R-squared, Within plus or minus 3 minutes, and Within plus or minus 5 minutes are higher-is-better measures.

The current test results identify XGBoost as the strongest model overall: MAE 3.879 minutes, RMSE 9.890 minutes, R-squared 0.0452, 58.80 percent within 3 minutes, and 84.88 percent within 5 minutes. Random Forest is second and Linear Regression is the baseline.

### Model error comparison

The grouped model-error bar chart visualizes MAE and RMSE for each model. Shorter bars indicate lower error. RMSE is normally higher than MAE because it gives more weight to large misses; a wide MAE-to-RMSE gap indicates that occasional large errors affect performance.

### Operational Risk Confusion Matrix

The Operational Risk Confusion Matrix groups actual and predicted taxi-out into Normal below 20 minutes, Moderate from 20 to below 30, High from 30 to below 45, and Severe at 45 minutes or more. Rows are actual bands, columns are predicted bands, and each cell is a test-flight count.

Large diagonal counts are correct band assignments. Off-diagonal counts are operational-category errors, with distant cells representing more serious band mismatches. This is a classification view of continuous regression predictions; the models were not trained as classifiers.

### Top Feature Importances for Random Forest

The feature-importance bar chart ranks the ten inputs most used by the fitted Random Forest. Longer bars mean greater impurity-based importance, and the displayed shares are loaded from the model-evaluation JSON. Current leaders include Distance, Departure Delay, scheduled-hour encodings, and congestion.

Importance indicates how much a feature helped split the training data. It does not show the direction of its effect and does not prove causation. Correlated or one-hot-encoded features may divide importance among themselves.

### Random Forest benchmark cards

Random Forest MAE and RMSE repeat that model's test errors and show their differences from the Linear Regression baseline. Negative error differences are improvements. Within plus or minus 3 and 5 minute cards show the share of Random Forest predictions whose absolute error is no greater than those tolerances; their deltas are percentage-point differences from baseline.

The Random Forest R-squared card shows its test R-squared and the numerical change from baseline. These cards are loaded from models/model_evaluation_metrics.json, so they remain aligned with the comparison table. A positive delta is desirable for tolerance accuracy and R-squared, while a negative delta is desirable for MAE and RMSE.

### Hub Airport Average Taxi-Out Times

The hub comparison chart displays configured reference averages: JFK 28.4 minutes, LGA 26.1, ORD 21.8, DFW 18.5, ATL 17.9, and DEN 16.2. The annotations describe broad operating contexts such as constrained geometry or primary project scope.

These values are currently hard-coded reference values in the dashboard, not live aggregations from the cleaned dataset. They provide comparative context only and should not be cited as current airport performance without external validation.

### Diurnal Departure Bank Wave Profile

The diurnal profile is a configured 24-hour reference curve of average taxi-out by hour. Higher points indicate the hours intended to represent departure-bank pressure; lower overnight points indicate lighter reference conditions.

The values are currently hard-coded in the dashboard rather than calculated from the loaded cleaned records. For evidence about this project's actual data, use the dynamic Mean Taxi-Out by Departure Hour chart or a live AI Analyst aggregation.

### Flights analysed

Flights analysed is the row count available to the Exploratory Analysis tab after required date, scheduled-time, departure-delay, and taxi-out fields are parsed and rows missing those required values are dropped. It is the denominator for the exploratory KPIs and may differ from another panel if its loading rules differ.

### Exploratory mean, median, 95th percentile, and Above 30 minutes

Mean taxi-out is the arithmetic average, while Median taxi-out is the middle observation and is less sensitive to long delays. The 95th percentile is the value at or below which 95 percent of analysed taxi-out observations fall; the remaining 5 percent are higher.

Above 30 minutes is the percentage of analysed flights whose taxi-out is greater than or equal to 30 minutes. In the exploratory code, HIGH_RISK uses TAXI_OUT greater than or equal to 30. This threshold combines the High and Severe operational risk bands and should not be confused with the separate congestion-density labels.

### Taxi-Out Distribution

The Taxi-Out Distribution is a 45-bin histogram. The horizontal axis is taxi-out minutes, and the height of each bar is the number of flights within that time interval. The dashed vertical line at 30 minutes marks the high-risk threshold.

The plot reveals the typical range, skewness, multiple peaks, and long-delay tail. It shows counts rather than probabilities, and the visual shape depends partly on the chosen bin width.

### Origin Airport Comparison

The Origin Airport Comparison ranks airports by mean taxi-out. Bar color represents the airport's 95th-percentile taxi-out, while hover information includes Flights, Median, and P95. Reading mean, tail value, and sample size together distinguishes a consistently slow airport from one affected by a small number of extreme cases.

### Congestion Heatmap by Day and Departure Hour

The heatmap groups records by weekday and scheduled departure hour, with color representing mean taxi-out minutes. Darker or hotter cells identify day-hour combinations associated with longer average taxi-out.

Despite the title, the color is not a direct count of CONGESTION_1H; it is taxi-out intensity used as a surface-congestion proxy. The chart is descriptive and should not be interpreted as proof that the day or hour caused the delay.

### Busiest Routes

The Busiest Routes chart ranks the 12 directional origin-destination pairs with the largest flight counts in the analysed sample. Bar length is Flights, color is mean taxi-out, and hover information includes P95 taxi-out.

Busiest therefore means most records in this dataset, not necessarily the highest real-world annual traffic. Route direction matters: origin A to destination B is different from origin B to destination A.

### High-Risk Departures by Hour

This bar chart groups flights by scheduled departure hour and shows the percentage with TAXI_OUT greater than or equal to 30 minutes. Hover information gives the total flights in each hour. A high bar identifies an hour with a larger observed share of long taxi-out events, but low-volume hours should be interpreted cautiously.

### Departure Delay versus Taxi-Out

The scatter plot places departure delay on the horizontal axis and taxi-out minutes on the vertical axis, with color identifying origin airport. Each point is a flight from a reproducible sample of up to 5,000 records after the optional origin filter.

The displayed correlation caption is calculated over the full exploratory dataset, not just the sampled points or selected-origin subset. Correlation measures linear association, not causation; a weak value supports treating departure delay and taxi-out as distinct operational signals.

### Flight Distance versus Departure Delay

This scatter plot places flight distance in miles on the horizontal axis and departure delay in minutes on the vertical axis. Each point is a flight from a reproducible sample of up to 5,000 records, colored by origin.

The chart helps identify broad relationships, clusters, and outliers. It does not show that distance causes departure delay, because both may be associated with route, airport, schedule, carrier, or operational conditions.

### Departure Delay by Distance Band

The box plot compares departure-delay distributions across the displayed distance categories. The line inside a box is the median, the box spans the interquartile range from P25 to P75, and the whiskers show the non-outlier spread under the plotting convention. Individual points are hidden.

The categories are 260-499, 500-999, 1,000-1,499, 1,500-1,999, and 2,000 or more miles in the current display. Wider boxes or longer whiskers indicate greater variability. Overlap between boxes means distance band alone does not cleanly separate departure-delay behavior.

### How to interpret all dashboard values safely

Counts describe how many records meet a condition. Means summarize arithmetic averages, medians summarize middle observations, and percentiles describe thresholds in a distribution. Percent values describe shares of a stated denominator, while percentage-point deltas are direct differences between percentages.

Model predictions and scenario values are estimates, descriptive charts summarize this sample, and configured reference charts are illustrative context. None of these values alone establishes causation or replaces current operational information such as weather, runway configuration, air-traffic-control instructions, or airline procedures.

## AI Analyst architecture

The AI Analyst uses a priority-based local architecture:

1. Normalize common spelling and phrase variations.
2. Answer calculation questions from the loaded flight data.
3. Answer project metrics from authoritative structured artifacts.
4. Match curated workbook questions and aliases.
5. Retrieve focused explanatory passages from approved Markdown knowledge sources.
6. Return the answer with sources and a confidence label.

The system uses local deterministic calculations, structured project artifacts, and TF-IDF document retrieval. It does not require an external language-model API.

## Live calculation capabilities

When the cleaned flight dataframe is available, the AI Analyst can calculate counts, shares, means, medians, percentiles, monthly summaries, airport rankings, carrier rankings, route rankings, distance summaries, departure-hour summaries, and taxi-out statistics for named airports, carriers, or routes.

Live calculations should take precedence over static documentation when the user asks for a value derived from the currently loaded records.

## Operational interpretation and A-CDM use

Taxi-out forecasts can support departure sequencing and gate-hold evaluation. Holding an aircraft at the gate during a predicted surface queue may reduce engine-running time on taxiways. Estimated wheels-off time can also improve coordination between airline dispatch, ramp operations, tower planning, and downstream flow management.

Recommendations remain advisory. Operational decisions require current weather, runway configuration, air-traffic-control instructions, airline procedures, and certified systems.

## Limitations

The project uses public historical data rather than live surveillance and operational feeds. It does not include real-time METAR weather, runway configuration, de-icing status, tug availability, gate conflicts, taxi-route geometry, runway-crossing queues, aircraft type, weight, controller instructions, or active traffic-flow restrictions.

The sample is not balanced equally across every airport and airline. Results for low-volume groups are less stable, and comparisons should include record counts.

Feature importance and model performance describe this dataset and implementation. They should not be generalized automatically to another airport, year, or operating environment.

## Terminology and question interpretation

The following expressions should be interpreted as equivalent when appropriate:

| User wording | Project meaning |
|---|---|
| best model, better model, top model, strongest algorithm | Model with the best out-of-time test metrics |
| RF, random forests, random forest model | Random Forest Regressor |
| XGB, boosted model, gradient boosting model | XGBoost Regressor |
| LR, baseline model, linear model | Linear Regression baseline |
| important features, top predictors, drivers, feature ranking | Model feature importance |
| confusion table, classification matrix, risk matrix | Operational risk confusion matrix |
| taxi time, ground time, pushback-to-takeoff time | Taxi-out duration when no competing metric is named |
| traffic, surface demand, departure queue | Congestion context |
| one-hour congestion, hourly pushbacks | `CONGESTION_1H` |
| half-hour congestion, recent pushbacks | `CONGESTION_30M` |
| accuracy within five minutes | Percentage of predictions within plus or minus 5 minutes |

When a question names a different metric such as departure delay, distance, flight duration, or model error, the AI Analyst should not silently reinterpret it as taxi-out time.

## Authoritative source order

For numerical facts, the preferred source order is:

1. Current loaded flight data for requested calculations.
2. `models/model_evaluation_metrics.json` for performance, hyperparameters, coefficients, and feature importance.
3. `data/processed/cleaning_audit_summary.json` for cleaning counts and taxi-out summary statistics.
4. `data/processed/ai_analyst_results.xlsx` for curated factual questions and aliases.
5. This knowledge document for definitions and explanations.

Planning notes, improvement checklists, bibliographic references, and example prose must not override the structured results above.
