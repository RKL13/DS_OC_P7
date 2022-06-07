# DS_OC_P7

### AWS Installation

```
sudo apt update
sudo apt install python3 python3-pip
sudo apt-get install python3-venv subversion tmux
```

##### Rest API

```
svn checkout https://github.com/RKL13/DS_OC_P7/trunk/rest_api_flask
cd rest_api_flask
chmod +x deploy.sh
./deploy.sh
```