# MLops_classifcation


[repo link]("https://github.com/umaretiya")

Create first flask classifciation  machine learning project
```{link of repo}
[repo link](https://github.com/umaretiya)
```

XGBoost model for credit card default prediction : ML projects
### Buld docker image
- Image name banking:latest
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
# cd to your newly created directory
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
