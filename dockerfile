FROM python:3.10.4 
RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib
RUN pip install tensorflow
RUN pip install Keras
RUN pip install scikit-learn
RUN pip install japanize-matplotlib 