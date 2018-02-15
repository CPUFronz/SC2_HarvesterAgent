# This script downloads and sets up the SC2 environment as well as a python virtual environment
# args path, tensorflow-gpu

GPU=false
INSTALL_DIR=$PWD

echo $INSTALL_DIR

for i in "$@"
do 
case $i in
    -dir=*)
    INSTALL_DIR="${i#*=}"
    ;;
    -gpu)
    GPU=true
    ;;
esac
done

cd $INSTALL_DIR

wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.3.17.zip
unzip -P iagreetotheeula SC2.3.17.zip
wget https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip
unzip mini_games.zip -d $INSTALL_DIR/StarCraftII/Maps

python3 -m venv venv
source $INSTALL_DIR/venv/bin/activate
pip install numpy==1.14.0
pip install PySC2==1.2
if $GPU
then
    pip install tensorflow-gpu=1.5.0
else
  pip install tensorflow==1.5.0
fi
pip install tflearn==0.3.2
pip install matplotlib==2.1.2

if grep -q "export SC2PATH=" $INSTALL_DIR/venv/bin/activate
  then
    sed -i "s|SC2PATH=.*|SC2PATH=$INSTALL_DIR/StarCraftII|" $INSTALL_DIR/venv/bin/activate
  else
    printf "\n#Installation path for StarCraft 2, needed for pysc2" >> $INSTALL_DIR/venv/bin/activate
    printf "\nexport SC2PATH=$INSTALL_DIR/StarCraftII\n" >> $INSTALL_DIR/venv/bin/activate
fi

#TODO: git clone
