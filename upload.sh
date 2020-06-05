#!/bin/bash
git init
git add .
git config user.name "hustwutao"
git config user.email hust.wutao@gmail.com
git remote add origin https://github.com/hustwutao/VAE.git
git commit -m "Updating Code"
git push --force -u origin master
