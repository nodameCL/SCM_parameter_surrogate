# 3 FC layers of DNN 
## architecture 1: BZ = 512, BN = True 
config = {'l1': 512, 'l2': 256, 'l3': 64, 'lr': 0.001, 
              'batch_size': 512, 'batch_norm' : True,
              'layer_norm' : False}
              
### res on test data: 
[250] train loss: 0.005146 
[250] validate loss: 0.018620
R2 of test is:  0.7496272544975173
Test set results for 11311 samples:
MSE: 0.019042173
MAE: 0.08170565
MSE loss on test set is: 3.86067265375425e-05
time used to train model with 20/250 patience is:  0.046019902076667576 hrs

### res on training data 
R2 of test is:  0.9240037814054232
Test set results for 90483 samples:
MSE: 0.00520012
MAE: 0.043627284
MSE loss on test set is: 1.0171274345690887e-05
time used to train model with 20/250 patience is:  0.0014235314116619218 hrs

## architecture 2: BZ = 256, BN = True 
config = {'l1': 512, 'l2': 256, 'l3': 64, 'lr': 0.001, 
              'batch_size': 256, 'batch_norm' : True,
              'layer_norm' : False}
### training procedure: 
[250] train loss: 0.005124 
[250] validate loss: 0.009855
R2 of test is:  0.8629873817417135
Test set results for 11311 samples:
MSE: 0.009769414
MAE: 0.06541492
MSE loss on test set is: 3.871559877228214e-05
time used to train model with 20/250 patience is:  0.0752588158430606 hrs

### res on training data: 
R2 of test is:  0.9263834776173414
Test set results for 90483 samples:
MSE: 0.005007508
MAE: 0.043545566
MSE loss on test set is: 1.9593338061919885e-05
time used to train model with 20/250 patience is:  0.0005134027350060125 hrs

## architecture 3: no constraint on C1, BZ = 256, BN = True 
- no constraint on C1 
- return self.fc4(x)
config = {'l1': 512, 'l2': 256, 'l3': 64, 'lr': 0.001, 
              'batch_size': 256, 'batch_norm' : True,
              'layer_norm' : False}
### training procedure               
R2 of test is:  0.7775074825741465
Test set results for 11311 samples:
MSE: 0.01644066
MAE: 0.082415834
MSE loss on test set is: 6.523955342977456e-05
time used to train model with 20/250 patience is:  0.06742363173111193 hrs  

## architecture 4: add layernorm 
### training procedure 
[250] train loss: 0.007182 
[250] validate loss: 0.007352
R2 of test is:  0.8938483663496719
Test set results for 11311 samples:
MSE: 0.007449708
MAE: 0.053439505
MSE loss on test set is: 2.9589119422879464e-05
time used to train model with 20/250 patience is:  0.06484391877611213 hrs

### res on training data 
R2 of test is:  0.9027950421257114
Test set results for 90483 samples:
MSE: 0.006761218
MAE: 0.050825667
MSE loss on test set is: 2.6441868863758664e-05
time used to train model with 20/250 patience is:  0.0005259392713924171 hrs

## architecture 5: add layernorm and batchnorm 
### training procedure 
              
> when adding constraint on C1, the best batch size is 256. Under the same batch size/architecture, performance: C1 w/ constraint > C1 w/o constraint 
> layernorm > batchnorm 

# architecture 5: 5 FC layers (including output layer)
- adding more layer: 24-512-512-256-64-5