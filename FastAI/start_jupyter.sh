[[ -d aicoc-ai-immersion ]] || git clone git@github.ibm.com:vanstee/aicoc-ai-immersion.git
nohup jupyter notebook --ip=0.0.0.0 --allow-root --port=5050 --no-browser --NotebookApp.token='aicoc' --NotebookApp.password='' &

