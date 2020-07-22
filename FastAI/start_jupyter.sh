# $1 is user_dir and user_port
user_dir=$1
user_port=$1

[[ -d $user_dir ]] || mkdir $user_dir; 
cd $user_dir

[[ -d aicoc-ai-immersion ]] || git clone git@github.ibm.com:vanstee/aicoc-ai-immersion.git
cd aicoc-ai-immersion
git fetch
git reset --hard origin/master
# kill existing jupyter
ps -ef |  grep -i [j]upyter-notebook.* | grep port=$1 | sed -e "s/ \{1,\}/ /g" | cut -d " " -f2 | xargs -i kill {}
# Startup 
nohup jupyter notebook --ip=0.0.0.0 --allow-root --port=$user_port --no-browser --NotebookApp.token='aicoc' --NotebookApp.password=''  &> classlog.out &

