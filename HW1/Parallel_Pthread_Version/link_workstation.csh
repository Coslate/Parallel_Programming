#! /bin/csh -f

#cd ./*/Testing/

#copy 'Testing' folder to workstation/Testing
scp -r -P 37106 ../Testing pp10767218@140.113.215.195:Testing

#link to workstation
ssh -p 37106 pp10767218@140.113.215.195

#show the number of processor
> cat /proc/cpuinfo | grep "processor" | wc -l

#show all the processors
> cat /proc/cpuinfo | grep "processor"
