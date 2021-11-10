# Homework 1: COVID-19 Cases Prediction (Regression)
Author: Heng-Jui Chang

Slides: https://github.com/ga642381/ML2021-Spring/blob/main/HW01/HW01.pdf
Videos (Mandarin): https://cool.ntu.edu.tw/courses/4793/modules/items/172854
https://cool.ntu.edu.tw/courses/4793/modules/items/172853
Video (English): https://cool.ntu.edu.tw/courses/4793/modules/items/176529


## Object
Solve a regression problem with DNN

## Description & task

![image](https://user-images.githubusercontent.com/69283174/141036190-c6324feb-c921-4ac9-ba39-e50224d9366a.png)

## Data

### Data analysis

```
 
id	AL	AK	AZ	AR	CA	CO	CT	FL	GA	ID	IL	IN	IA	KS	KY	LA	MD	MA	MI	MN	MS	MO	NE	NV	NJ	NM	NY	NC	OH	OK	OR	PA	RI	SC	TX	UT	VA	WA	WV	WI	cli	ili	hh_cmnty_cli	nohh_cmnty_cli	wearing_mask	travel_outside_state	work_outside_home	shop	restaurant	spent_time	large_event	public_transit	anxious	depressed	felt_isolated	worried_become_ill	worried_finances	tested_positive	cli	ili	hh_cmnty_cli	nohh_cmnty_cli	wearing_mask	travel_outside_state	work_outside_home	shop	restaurant	spent_time	large_event	public_transit	anxious	depressed	felt_isolated	worried_become_ill	worried_finances	tested_positive	cli	ili	hh_cmnty_cli	nohh_cmnty_cli	wearing_mask	travel_outside_state	work_outside_home	shop	restaurant	spent_time	large_event	public_transit	anxious	depressed	felt_isolated	worried_become_ill	worried_finances	tested_positive
0	1.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.81461	0.7713562	25.6489069	21.2420632	84.6446717	13.4624747	36.519841	63.1390944	23.8351187	44.7260552	16.9469288	1.7162617	15.4941927	12.0432752	17.0006473	53.4393163	43.279629	19.586492	0.8389952	0.8077665	25.6791006	21.2802696	84.005294	13.4677158	36.6378869	63.3186499	23.6888817	44.3851661	16.4635514	1.664819	15.2992283	12.0515055	16.5522637	53.2567949	43.6227275	20.1518381	0.8978015	0.8878931	26.0605436	21.5038315	84.4386175	13.0386108	36.4291187	62.4345385	23.8124113	43.4304231	16.1515266	1.602635	15.4094491	12.0886885	16.7020857	53.9915494	43.6042293	20.7049346
1	1.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.8389952	0.8077665	25.6791006	21.2802696	84.005294	13.4677158	36.6378869	63.3186499	23.6888817	44.3851661	16.4635514	1.664819	15.2992283	12.0515055	16.5522637	53.2567949	43.6227275	20.1518381	0.8978015	0.8878931	26.0605436	21.5038315	84.4386175	13.0386108	36.4291187	62.4345385	23.8124113	43.4304231	16.1515266	1.602635	15.4094491	12.0886885	16.7020857	53.9915494	43.6042293	20.7049346	0.9728421	0.9654959	25.7540871	21.0162096	84.1338727	12.5819525	36.4165569	62.0245166	23.6829744	43.1963133	16.1233863	1.641863	15.230063	11.8090466	16.5069733	54.185521	42.6657659	21.2929114
```
+ 40 states（四十个州）：AL	AK	AZ	AR	CA	CO	CT	FL	GA	ID	IL	IN	IA	KS	KY	LA	MD	MA	MI	MN	MS	MO	NE	NV	NJ	NM	NY	NC	OH	OK	OR	PA	RI	SC	TX	UT	VA	WA	WV	WI
+ 4 illness（四个疑似症状）:cli	ili	hh_cmnty_cli	nohh_cmnty_cli
+ 8 behavior indicators(8个行为）：	wearing_mask	travel_outside_state	work_outside_home	shop	restaurant	spent_time	large_event	public_transit
+ 5 mental health(5个精神状况）：anxious	depressed	felt_isolated	worried_become_ill	worried_finances	
+ tested positive: 是否确诊
### data description
one hot vectors： 用一个 维度50的向量 来表示每个特性，sample 满足这个特性时， 该位置值为1
![image](https://user-images.githubusercontent.com/69283174/141039703-e4bf2724-f0a5-4497-80c1-596e077a106d.png)


# Hints

## Simple Baseline 

Run sample code

## Medium Baseline 
  Feature selection: 40 states + 2 `tested_positive` (`TODO` in dataset)

## Strong Baseline 
1. Feature selection (what other features are useful?)
2. DNN architecture (layers? dimension? activation function?)
3. Training (mini-batch? optimizer? learning rate?)
4. L2 regularization
5. There are some mistakes in the sample code, can you find them?
