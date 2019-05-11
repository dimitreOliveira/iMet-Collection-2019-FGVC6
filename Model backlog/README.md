## Model backlog (list of developed model and it's score)
- **Train** and **validation** are the splits using the train data from the competition.
- The competition metric is **Mean F-Score Beta**.
- **Runtime** is the time in seconds that the kernel took to finish.
- **Pb Leaderboard** is the Public Leaderboard score.
- **Pv Leaderboard** is the Private Leaderboard score.

---

## Deep Learning

### Complete models

|Model|Train|Validation|Pb Leaderboard|Pv Leaderboard|Runtime(s)|
|-----|-----|----------|--------------|--------------|----------|
|[1st] - CNN cat_loss - iMet|???|???|0.089|???|456|
|[2nd] - CNN bin_loss - iMet|???|???|0.008|???|689|
|[3rd] - CNN cat_loss Sft-max - iMet|???|???|0.008|???|381.5|
|[4th] - CNN bin_loss thrs- iMet|???|???|0.022|???|553.6|
|[5th] - CNN small model - iMet|???|???|0.032|???|1082.2|
|[6th] - CNN F2 metric - iMet|???|???|0.026|???|1039.1|
|[7th] - DL threshold finder|???|???|0.068|???|1056.2|
|[8th] - DL - LR Scheduler|???|???|0.064|???|21015.9|
|[9th] - DL - Bigger model|???|???|0.161|???|29268.7|
|[10th] - DL - Data augmentation|???|???|0.12|???|16601.4|

### Transfer Learning

#### VGG16
    
|Model|Train|Validation|Pb Leaderboard|Pv Leaderboard|Runtime(s)|
|-----|-----|----------|--------------|--------------|----------|
|[11th] - Bottleneck VGG16|???|???|0.167|???|2349.9|
|[12th] - Bottleneck VGG16|???|???|0.168|???|2759.3|
|[13th] - Bottleneck VGG16 - SftMax|???|???|0.13|???|2095.3|
|[14th] - Bottleneck VGG16 img128|???|???|0.225|???|2897.4|
|[15th] - Bottleneck VGG16 img256|???|???|???|???|???|
|[16th] - Bottleneck VGG16 - Adam|???|???|0.422|???|1465.3|
|[17th] - Bottleneck VGG16 - Big1|???|???|0.21|???|2613.5|
|[18th] - Bottleneck VGG16 - Big2|???|???|0.246|???|3068.4|
|[21th] - Bottleneck VGG16 - Adam - Big1|???|???|0.456|???|1675.6|
|[22th] - Bottleneck VGG16 - Adam - Big2|???|???|0.457|???|1742.3|
|[23th] - Bottleneck VGG16 - Adam - Big3|???|???|0.474|???|1828|
|[24th] - Bottleneck VGG16 - Adam - Big4|???|???|0.467|???|2320.4|
|[26th] - Bottleneck VGG16 - LR 0.001|???|???|0.378|???|1493|
|[27th] - Bottleneck VGG16 - LR 0.00001|???|???|0.463|???|3425.8|
|[28th] - Bottleneck VGG16 - BatchNorm|???|???|0.419|???|1578.8|
|[30th] - Bottleneck VGG16 - GlobalAVG|???|???|0.377|???|3310.6|
|[31th] - Fine-tune - VGG16 - Complete|???|???|0.454|???|12760.2|
|[32th] - Fine-tune - VGG16 - Top 2 conv|???|???|0.445|???|12307.6|
|[33th] - Fine-tune - VGG16 - Top conv|???|???|0.45|???|15714.9|
|[34th] - Fine-tune - VGG16 - head|???|???|0.297|???|4621.5|
|[35th] - Fine-tune - VGG16 - Top Conv 2|???|???|0.279|???|7078.5|
|[45th] - Fine-tune - VGG16 - Top conv|???|???|0.247|???|5478|
|[54th] - Fine-tune - VGG16 - head2|???|???|0.319|???|10139.6|
|[63th] - Fine-tune - VGG16 - Top conv2|???|???|0.249|???|16400.4|
|[71th] - Fine-tune - VGG16 - Top 2 conv2|???|???|0.214|???|17680.4|
|[80th] - Fine-tune - VGG16 - Complete|???|0.266|0.251|???|16923.6|
|[89th] - Fine-tune - VGG16 - Complete Adam|???|0.438|0.445|???|16688.1|
|[98th] - Fine-tune - VGG16 - Complete Adam2|???|0.452|0.457|???|13884.2|

#### VGG19
    
|Model|Train|Validation|Pb Leaderboard|Pv Leaderboard|Runtime(s)|
|-----|-----|----------|--------------|--------------|----------|
|[19th] - Bottleneck VGG19|???|???|0.215|???|3493.4|
|[25th] - Bottleneck VGG19 - Adam - Big|???|???|0.457|???|1715.5|
|[36th] - Fine-tune - VGG19 - head|???|???|0.29|???|4616.7|
|[46th] - Fine-tune - VGG19 - Top conv|???|???|0.294|???|8585.7|
|[55th] - Fine-tune - VGG19 - head2|???|???|0.313|???|9212.1|
|[64th] - Fine-tune - VGG19 - Top conv2|???|???|0.318|???|24458.9|
|[72th] - Fine-tune - VGG19 - Top 2 conv2|???|???|0.321|???|24829.4|
|[81th] - Fine-tune - VGG19 - Complete|???|0.329|0.321|???|25104.2|
|[90th] - Fine-tune - VGG19 - Complete Adam|???|0.440|0.445|???|31285|
|[99th] - Fine-tune - VGG19 - Complete Adam2|???|0.447|0.450|???|17153.8|


#### InceptionV3
    
|Model|Train|Validation|Pb Leaderboard|Pv Leaderboard|Runtime(s)|
|-----|-----|----------|--------------|--------------|----------|
|[20th] - Bottleneck InceptionV3|???|???|0.29|???|3041.3|
|[29th] - Bottleneck InceptionV3-Adam Big|???|???|0.412|???|1481|
    
#### Xception
    
|Model|Train|Validation|Pb Leaderboard|Pv Leaderboard|Runtime(s)|
|-----|-----|----------|--------------|--------------|----------|
|[38th] - Fine-tune - Xception - head|???|???|0.22|???|7076.7|
|[47th] - Fine-tune - Xception - Top conv|???|0.212|0.212|???|11659.3|
|[56th] - Fine-tune - Xception - head2|???|???|0.235|???|15754.1|
|[65th] - Fine-tune - Xception - Top conv2|???|???|0.212|???|12202.3|
|[73th] - Fine-tune - Xception - Top 2 conv2|???|???|0.22|???|29792.8|
|[82th] - Fine-tune - Xception - Complete|???|0.310|0.298|???|31066.4|
|[91th] - Fine-tune - Xception - Complete Adam|???|0.459|0.465|???|30803.3|
|[100th] - Fine-tune - Xception - Complete Adam2|???|0.451|0.457|???|12655.8|

#### ResNet50
    
|Model|Train|Validation|Pb Leaderboard|Pv Leaderboard|Runtime(s)|
|-----|-----|----------|--------------|--------------|----------|
|[39th] - Fine-tune - ResNet50 - head|???|???|0.186|???|4666.7|
|[48th] - Fine-tune - ResNet50 - Top conv|???|???|0.185|???|9309.8|
|[57th] - Fine-tune - ResNet50 - head2|???|???|0.185|???|9246.8|
|[66th] - Fine-tune - ResNet50 - Top conv2|???|???|0.185|???|10064.4|
|[74th] - Fine-tune - ResNet50 - Top 2 conv 2|???|???|0.184|???|9985.1|
|[83th] - Fine-tune - ResNet50 - Complete|???|0.381|0.383|???|26435.1|
|[92th] - Fine-tune - ResNet50 - Complete Adam|???|0.459|0.472|???|23974.9|
|[101th] - Fine-tune - ResNet50 - Complete Adam2|???|0.460|0.466|???|13656|

#### MobileNetV2

|Model|Train|Validation|Pb Leaderboard|Pv Leaderboard|Runtime(s)|
|-----|-----|----------|--------------|--------------|----------|
|[40th] - Fine-tune - MobileNetV2 - head|???|???|0.268|???|5486.2|
|[49th] - Fine-tune - MobileNetV2 - Top conv|???|???|0.265|???|9675.5|
|[58th] - Fine-tune - MobileNetV2 - head2|???|???|0.265|???|10023.7|
|[67th] - Fine-tune - MobileNetV2 - Top conv2|???|???|0.267|???|23489.4|
|[75th] - Fine-tune - MobileNetV2 - Top 2 conv 2|???|???|0.224|???|25650.3|
|[84th] - Fine-tune - MobileNetV2 - Complete|???|0.391|0.39|???|27602.8|
|[93th] - Fine-tune - MobileNetV2 - Complete Adam|???|0.489|0.495|???|26762.7|
|[102th] - Fine-tune - MobileNetV2 - Complete Adam2|???|0.476|0.480|???|17248.3|

#### DenseNet121
    
|Model|Train|Validation|Pb Leaderboard|Pv Leaderboard|Runtime(s)|
|-----|-----|----------|--------------|--------------|----------|
|[41th] - Fine-tune - DenseNet121 - head|???|???|0.263|???|9256|
|[50th] - Fine-tune - DenseNet121 - Top conv|???|???|0.268|???|14319.9|
|[59th] - Fine-tune - DenseNet121 - head2|???|???|0.268|???|16165.2|
|[68th] - Fine-tune - DenseNet121 - Top conv2|???|???|0.268|???|22702.6|
|[76th] - Fine-tune - DenseNet121 - Top 2 conv 2|???|???|0.27|???|32002.6|
|[85th] - Fine-tune - DenseNet121 - Complete|???|0.365|0.362|???|26745.2|
|[94th] - Fine-tune - DenseNet121 - Complete Adam|???|0.475|0.481|???|32304|
|[103th] - Fine-tune - DenseNet121 - Complete Adam2|???|0.470|0.476|???|32304|

#### DenseNet169
    
|Model|Train|Validation|Pb Leaderboard|Pv Leaderboard|Runtime(s)|
|-----|-----|----------|--------------|--------------|----------|
|[42th] - Fine-tune - DenseNet169 - head|???|???|0.251|???|10790|
|[51th] - Fine-tune - DenseNet169 - Top conv|???|???|0.258|???|14111.5|
|[60th] - Fine-tune - DenseNet169 - head2|???|???|0.25|???|13841.3|
|[69th] - Fine-tune - DenseNet169 - Top conv2|???|???|0.257|???|28712|
|[77th] - Fine-tune - DenseNet169 - Top 2 conv 2|???|???|0.256|???|28374.3|
|[86th] - Fine-tune - DenseNet169 - Complete|???|0.377|0.371|???|30590.9|
|[95th] - Fine-tune - DenseNet169 - Complete Adam|???|0.480|0.486|???|24821.5|
|[104th] - Fine-tune - DenseNet169 - Comple Adam2|???|0.480|0.490|???|25554.1.5|

#### NasNetLarge
    
|Model|Train|Validation|Pb Leaderboard|Pv Leaderboard|Runtime(s)|
|-----|-----|----------|--------------|--------------|----------|
|[43th] - Fine-tune - NasNetLarge - head|???|???|0.197|???|9221.4|
|[52th] - Fine-tune - NasNetLarge - Top conv|???|???|0.201|???|12249.5|
|[61th] - Fine-tune - NasNetLarge - head2|???|???|0.194|???|14660.6|
|[70th] - Fine-tune - NasNetLarge - Top conv2|???|???|0.198|???|25511.2|
|[78th] - Fine-tune - NasNetLarge - Top 2 conv 2|???|???|0.198|???|26121.9|
|[87th] - Fine-tune - NasNetLarge - Complete|???|0.321|0.314|???|20585.2|
|[96th] - Fine-tune - NasNetLarge - Complete Adam|???|0.464|0.478|???|28834.8|
|[105th] - Fine-tune - NasNetLarge - Complete Adam2|???|0.468|0.481|???|31973.1.8|

#### NasNetMobile

|Model|Train|Validation|Pb Leaderboard|Pv Leaderboard|Runtime(s)|
|-----|-----|----------|--------------|--------------|----------|
|[44th] - Fine-tune - NasNetMobile - head|???|???|0.211|???|3765.2|
|[53th] - Fine-tune - NasNetMobile - Top conv|???|???|0.215|???|13457|
|[62th] - Fine-tune - NasNetMobile - head2|???|???|0.213|???|8930.6| 
|[71th] - Fine-tune - NasNetMobile - Top conv2|???|???|0.214|???|26632| 
|[79th] - Fine-tune - NasNetMobile - Top 2 conv 2|???|???|0.212|???|24337|
|[88th] - Fine-tune - NasNetMobile - Complete|???|0.296|0.291|???|29064.1|
|[97th] - Fine-tune - NasNetMobile - Complet Adam|???|0.461|???|???|30691.5|
|[106th] - Fine-tune - NasNetMobile - CompleteAdam2|???|0.445|0.448|???|20915.5|
