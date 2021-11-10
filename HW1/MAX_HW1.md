# Homework 1: COVID-19 Cases Prediction (Regression)
 

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
### Data description
one hot vectors： 用一个 向量 来表示每个特性，sample 满足这个特性时， 该位置值为1
![image](https://user-images.githubusercontent.com/69283174/141039703-e4bf2724-f0a5-4497-80c1-596e077a106d.png)
training data: 矩阵维度是2700×94, 2700是sample的个数， 94是所有的属性，其中前40是州，18×3是3天的症状和行为特性和一个确诊。
![image](https://user-images.githubusercontent.com/69283174/141040812-785d465c-d60d-4dd7-bfe8-0bc7f5585b28.png)
testing data: 矩阵维度是893×93, 893是sample的个数， 93是所有的属性，其中前40是州，18×2是2天的所有症状和行为特性和一个确诊。最后一天是17个属性，因为确诊是否是需要预测出来的。
![image](https://user-images.githubusercontent.com/69283174/141040812-785d465c-d60d-4dd7-bfe8-0bc7f5585b28.png)

## Code

### COVID19Dataset(Dataset) 

```
class COVID19Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self,
                 path,
                 mode='train',
                 target_only=False):
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)
        
        if not target_only:
            feats = list(range(93))
        else:
            # TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
            pass

        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            data = data[:, feats]
            
            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            
            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)
 ```


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
