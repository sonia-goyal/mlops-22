# mlops-22
Repo for mlops class July 2022
```
Run SVM  Decision Tree
1   0.97  0.88
2   0.98  0.88
3   0.98  0.81
4   1.0   0.89
5   0.99  0.83
mean:  0.98   0.86
std:  0.01   0.03
```

Add flask app
Command:

```
export FLASK_APP=app; flask run
```

Docker image name for flask app

```
REPOSITORY                     TAG                                IMAGE ID       CREATED          SIZE
exp                            flask_app                          6cbad1f72377   5 minutes ago    437MB
<none>                         <none>                             61c1457c06d1   6 minutes ago    437MB
```

Dockerfile Name: Dockerfile_flask
Image name: flask_app - This image is using separate requirements file which have the necessary packages only.
Exposed port: 5050

