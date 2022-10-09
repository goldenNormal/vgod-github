# Are we really making much progress in unsupervised graph outlier detection? Revisiting the problem with new insight and method

This is the source code of  paper ”Are we really making much progress in unsupervised graph outlier detection? Revisiting the problem with new insight and method“

![VGOD-framework](./fig/VGOD-framework.png)



## Requirements

This code requires the following:

- Python>=3.7
- PyTorch>=1.12.1
- Numpy>=1.19.2
- Scipy>=1.6.2
- Scikit-learn>=0.24.1
-  PyG  >= 2.1.0

## Running the experiments

### OD experiment

#### step 1: outlier injection

 This is a pre-processing step which injects outliers into the original clean datasets. Take Cora dataset as an example: 

```
python inject_anomaly.py --dataset cora
```

 After outlier injection, the disturbed datasets are saved into "data" folder 

#### step 2: outlier detection

 This step is to run the **VGOD** framework to detect outliers in the network datasets. Take Cora dataset as an example: 

```
python train_sep.py --data cora
```



### Robustness of structural outlier detection experiment

#### step 1: outlier injection

 This is a pre-processing step which injects outliers into the original clean datasets. 

```
python struct_ano_detect.py
```

 After anomaly injection, the disturbed datasets are saved into "struct_datasets" folder 

#### step 2: outlier detection

 This step is to run the **VBM** to detect outliers in the network datasets. 

```
python struct_ano_detect.py --data Cora
```






