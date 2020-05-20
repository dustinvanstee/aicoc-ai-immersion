conda install -y spacy
conda install -y pytorch
conda install -y jupyter

conda install -y pandas

## Install FastAI dependencies

conda install -y -c anaconda nltk
conda install -y bottleneck
conda install -y beautifulsoup4
conda install -y numexpr
conda install -y nvidia-ml-py3 -c fastai
conda  install -y  packaging
pip install --no-deps fastai
pip install  dataclasses
pip install fastprogress

# CECC only 
sudo yum install -y libxml2-devel libxslt-devel
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

