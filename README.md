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
```

##### Client

```
svn checkout https://github.com/RKL13/DS_OC_P7/trunk/client_streamlit
cd client_streamlit
```

##### Then

```
chmod +x deploy.sh
tmux new -s session_1
./deploy.sh
ctrl-b, d
```