#!bash/sh
epsilon=0.005

if ["$1" == ""]
then
netname="3_10"
else
netname=$1
fi

if ["$2" == ""]
then
imgname="46"
else
imgname=$2
fi
echo $netname $imgname

if [$netname == "-all"] && [$imgname == "-all"]
then
    for net in $(ls ../mnist_nets)
        do
        echo $net | tee -a all_on_all.log
        for img in $(ls ../mnist_images)
            do
            echo $img | tee -a all_on_all.log
            python3 analyzer.py ../mnist_nets/$net ../mnist_images/$img $epsilon | tee -a all_on_all.log
        done
    done
fi

if ["$imgname" == "-all"]
then
    echo net $netname
    for img in $(ls ../mnist_images)
        do
        echo $img | tee -a all_img_on_net$netname.log
        python3 analyzer.py ../mnist_nets/mnist_relu_$netname.txt ../mnist_images/$img $epsilon | tee -a all_img_on_net$netname.log
    done
fi

if ["$netname" == "-all"]
then
    echo image $imagename
    for net in $(ls ../mnist_nets)
    do
        echo $netname | tee -a all_net_on_img$imgname.log
        python3 analyzer.py ../mnist_nets/$net ../mnist_images/img$imgname.txt $epsilon | tee -a all_net_on_img$imgname.log
    done
else
    echo image $imgname with net $netname
    python3 analyzer.py ../mnist_nets/mnist_relu_$netname.txt ../mnist_images/img$imgname.txt $epsilon 
fi

