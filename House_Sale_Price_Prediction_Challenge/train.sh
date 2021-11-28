#!/bin/bash
sudo apt-get install python3.7
python --version

sudo apt install python3-pip
pip3 --version

pip3 install pandas
pip3 install numpy 
pip3 install tensorflow
pip3 install matplotlib
pip3 install seaborn
pip3 install keras
pip3 install scikit-learn
pip3 install mlxtend

python3 ml_house_sale_price_prediction.py
