FILE=$1

if [[ $FILE != "facades" && $FILE != "deepfashion" && $FILE != "celeba" && $FILE != "ade20k" && $FILE != "cityscapes" && $FILE != "coco_stuff" ]]; 
	then echo "Available datasets are facades, deepfashion, celeba, cityscapes, ade20k, coco_stuff"
	exit 1
fi


echo "Specified [$FILE]"

URL=http://disi.unitn.it/~hao.tang/uploads/datasets/DAGAN/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE
