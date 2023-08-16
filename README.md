# ML_IoT_NIDs
Exploring IoT security with ML models. Detect vulnerabilities, analyze trade-offs, and reduce dimensions using autoencoders. Join us in fortifying IoT networks. 🛡️
### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- 


### Run

In a terminal or command window, navigate to the project directory `ML_ToN/` (that contains this README) and run one of the following commands:

```bash
python Supervised.py
```  
or
```bash
python Unsupervised.py
```
or open with Pycharm 
```bash
Run 
Supervised.py
Unsupervised.py


For Data Preprocessing
Open Data_Preprocessing.py and run code. This cleans data with label encoding of categorical features and save the preprcessed data to Clean.csv file 
this code splits the data set for unsupervised training since it is trained with only one class of the data.
Auto_encoder .py returns the model that reduces the dimension of dataset, Note to specify either supervised data or unsupervised data to train the autoencoder
