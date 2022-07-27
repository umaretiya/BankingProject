# ML_classifcation :Credit Default predictions-Banking Sector


[git link]("https://github.com/umaretiya/BankingProject")
[git repo link]("https://github.com/umaretiya/BankingProject.git")
[heroku app-link]("https://ml-classify.herokuapp.com/")
Create first flask classifciation  machine learning project
```{link of repo}
[repo link](https://github.com/umaretiya)
```
conda create -p venv python==3.7 -y
ml-classify

XGBoost model for credit card default prediction : ML projects
### Buld docker image
- Image name banking:latest
```
docker build -t <image name>:<tag name> .
```
python -m pip install --upgrade pip'

docker image flask_app and credit_card
docker image build -t <flask_docker> . # credit_card  //for docker <doker_banking>
docker run -p 5000:5000 -d <flask_docker> 
docker run --name flask1 -dit -p 5000:5000 a5f9c7a9a19f
docker run -p 5000:5000 -e PORT=5000 a5f9c7a9a19f

docker ps
docker stop <container_id>
docker login
##### renaming docker images
credit_card
docker tag flask_docker <your-docker-hub-username>/<flask-docker>

docker push <your-docker-hub-username>/<flask-docker>
docker images
heroku login
docker login --username=<your-username> --password=<your-password>
heroku create <app-name>
heroku container:push web --app <app-name>
heroku container:release web --app <app-name>
> Note: imagename fro docker must be lowercase

heroku container:push web -a <name heroku app>
heroku container:release web -a <name heroku app>
heroku open -a <name heroku app>
heroku logs --tail -a <name heroku app>


To list docker image
```
docker images
```
Run docker image
```
docker run -p 5000:5000 -e PORT=5000 f3322f5e3b00
```
python -c 'import secrets; print(secrets.token_hex())'

To check runnig container in docker
```
docker ps
```

to stop docker container
```
docker stop <container_id> ad6cabc7a281
```
Creating yaml file
```
.github/workflows/main.yaml
```
### Git commands:
echo "# BankingProject" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/umaretiya/BankingProject.git
git push -u origin main

###
git config --global core.compression 0
git clone --depth 1 <repo_URI>
##### cd to your newly created directory
git fetch --unshallow 
git pull --all

### Buld docker image
```
docker build -t <image name>:<tag name> .
```
> Note: imagename fro docker must be lowercase

To list docker image
```
docker images
```
Run docker image
```
docker run -p 5000:5000 -e PORT=5000 6ce17fe3d920
```

To check runnig container in docker
```
docker ps
```

to stop docker container
```
docker stop <container_id> ad6cabc7a281
```
Creating yaml file
```
.github/workflows/main.yaml
```
```
python setup.py install
```
install ipykernel
```
pip install ipykernel
```


### Final ramarks: ML Clssification projects
- docker images - banking
- get repo - MLops_classifcation
- heroku app - ml-classify
- local dir - MLops_classifcation
- Author - Keshav
- Sector - Banking
- Use case - Default of Credit Card clients of Bank
- Datasets - archive.ics.uci.edu/ml/datasets
- Final Data - kaggle - default-of-credit-card-clients-dataset
- labels 0 or 1 - 1 for default and 0 for not default
- Model - XGBClassifier
- Accuracy - 81 %
- Framework - Flask-Python
- environment - conda
