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
|[107th] - Fine-tune - VGG16 - Complete SGD|???|0.242|0.227|???|5194.9|
|[116th] - Fine-tune - VGG16 - Complete 2 Opt|???|0.447|0.449|???|12333.7|
|[125th] - Fine-tune - VGG16 - Complete 2 Opt2|???|0.454|0.457|???|12635.2|
|[126th] - Fine-tune - VGG16 - Complete Adam 128|???|0.479|0.482|???|13591.52|
|[135th] - Fine-tune - VGG16 - Original size|???|0.490|0.490|???|121595.2|

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
|[108th] - Fine-tune - VGG19 - Complete SGD|???|0.291|0.279|???|7914.4|
|[117th] - Fine-tune - VGG19 - Complete 2 Opt|???|0.443|0.444|???|17036.3|
|[127th] - Fine-tune - VGG19 - Complete Adam 128|???|0.467|0.477|???|18643.6|
|[136th] - Fine-tune - VGG19 - Original size|???|0.478|0.481|???|22742.4|


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
|[109th] - Fine-tune - Xception - Complete SGD|???|0.244|0.226|???|8630.9|
|[118th] - Fine-tune - Xception - Complete 2 Opt|???|0.447|0.454|???|18949|
|[128th] - Fine-tune - Xception - Complete Adam 128|???|0.498|0.504|???|20561.2|
|[137th] - Fine-tune - Xception - Original size|???|0.417|0.418|???|25167.6|
|[152th] - Xception - WarmUp|???|0.489|0.494|???|11150.6|
|[153th] - Xception - Batch 64|???|0.527|0.537|???|10745.8|
|[155th] - Xception - Preprocess - HFlip|???|0.535|0.546|???|12822.6|
|[156th] - Xception - Preprocess - HFlip2|???|0.536|0.547|???|17159.5|
|[158th] - Xception - 299x299|???|0.407|0.405|???|3497.6|
|[162th] - Xception - Preprocess - 2048 Head|???|0.514|0.521|???|14698.2|
|[163th] - Xception - Preprocess - 1024 Head|???|0.536|0.548|???|14553.2|
|[164th] - Xception - Preprocess - GbAvgPool2D|???|0.536|0.544|???|13364.3|
|[165th] - Xception - Preprocess - GbAvgPoo2D Drop|???|0.539|0.549|???|14961|
|[167th] - Xception - Preprocess - GbAvgPool2D|???|0.515|0.522|???|13829.2|
|[168th] - Xception - Monitor F2 0.15|???|0.528|0.545|???|21656.3|
|[169th] - Xception - Monitor F2 0.5|???|0.533|0.523|???|13909.1|

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
|[110th] - Fine-tune - ResNet50 - Complete SGD|???|0.358|0.352|???|8629.6|
|[119th] - Fine-tune - ResNet50 - Complete 2 Opt|???|0.456|0.461|???|16717.7|
|[129th] - Fine-tune - ResNet50 - Complete Adam 128|???|0.471|0.474|???|14285|
|[138th] - Fine-tune - ResNet50 - Original size|???|0.439|0.434|???|10461.8|
|[154th] - ResNet50 - Original|???|0.504|0.507|???|15972.1|
|[157th] - ResNet50 - Preprocess HFlip|???|0.515|0.518|???|21516.5|
|[159th] - ResNet50 - 2048 Head|???|0.522|0.528|???|25336.5|
|[160th] - ResNet50 - 1024 Head|???|0.532|0.540|???|19384.3|
|[161th] - ResNet50 - GbAvgPool2D|???|0.473|0.477|???|13972.8|
|[166th] - ResNet50 - GbAvgPool2D Drop|???|0.527|0.535|???|17896.3|

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
|[111th] - Fine-tune - MobileNetV2 - Complete SGD|???|0.378|0.378|???|11977.1|
|[120th] - Fine-tune - MobileNetV2 - Complete 2 Opt|???|0.470|0.475|???|22509.4|
|[130th] - Fine-tune - MobileNetV2 - Complete Adam 128|???|0.480|0.484|???|11453.8|
|[139th] - Fine-tune - MobileNetV2 - Original size|???|0.490|0.495|???|19120.6|
|[144th] - MobileNetV2 - Original Size - Warmup|???|0.494|0.497|???|14400.8|
|[145th] - MobileNetV2 - Original Size - WarmUp2|???|0.384|0.371|???|29845.1|
|[146th] - MobileNetV2 - Original Size - WarmUp3|???|0.373|0.362|???|12288.4|
|[147th] - MobileNetV2 - Preprocess|???|0.420|0.414|???|17704.4|
|[148th] - MobileNetV2 - Preprocess - HFlip|???|0.489|0.484|???|10250.4|
|[149th] - MobileNetV2 - Adam RLR|???|0.502|0.502|???|16004.3|

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
|[112th] - Fine-tune - DenseNet121 - Complete SGD|???|0.366|0.363|???|17562.8|
|[121th] - Fine-tune - DenseNet121 - Complete 2 Opt|???|0.470|0.479|???|28894.8|
|[131th] - Fine-tune - DenseNet121 - Complete Adam 128|???|0.494|0.5|???|17732.5|
|[140th] - Fine-tune - DenseNet121 - Original size|???|0.393|0.386|???|11661.8|

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
|[104th] - Fine-tune - DenseNet169 - Comple Adam2|???|0.480|0.490|???|25554.1|
|[113th] - Fine-tune - DenseNet169 - Complete SGD|???|0.372|0.366|???|18395.9|
|[122th] - Fine-tune - DenseNet169 - Complete 2 Opt|???|0.460|0.468|???|26055.6|
|[132th] - Fine-tune - DenseNet169 - Complete Adam 128|???|0.498|0.507|???|20004.2|
|[141th] - Fine-tune - DenseNet169 - Original size|???|0.405|0.404|???|12160|
|[150th] - DenseNet169 - Complete Adam 128 - 2|???|0.405|0.512|???|28314.8|
|[151th] - DenseNet169 - WarmUp|???|0.505|0.512|???|14885.4|

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
|[105th] - Fine-tune - NasNetLarge - Complete Adam2|???|0.468|0.481|???|31973.1|
|[115th] - Fine-tune - NasNetLarge - Complete SGD|???|0.321|0.314|???|21982.9|
|[124th] - Fine-tune - NasNetLarge - Complete 2 Opt|???|0.409|0.409|???|23384.3|
|[133th] - Fine-tune - NasNetLarge - Complete Adam 128|???|0.325|0.315|???|14925.4|
|[142th] - Fine-tune - NasNetLarge - Original Size|???|0.500|0.409|???|31045.7|

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
|[114th] - Fine-tune - NasNetMobile - Complete SGD|???|0.272|0.261|???|15163.7|
|[123th] - Fine-tune - NasNetMobile - Complete 2 Op|???|0.423|0.425|???|22051.6|
|[134th] - Fine-tune - NasNetMobile - Complete Adam 128|???|0.467|0.437|???|27208.3|
|[143th] - Fine-tune - NasNetMobile - Original Size|???|0.492|0.501|???|27742.4|
