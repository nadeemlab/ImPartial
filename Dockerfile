FROM projectmonai/monailabel:latest

ENV APP_HOME /opt/monai

# RUN git clone https://github.com/nadeemlab/ImPartial.git $APP_HOME

COPY impartial/ $APP_HOME/impartial/
COPY monailabel-app/ $APP_HOME/monailabel-app/

COPY ./setup.py $APP_HOME/setup.py
COPY ./README.md $APP_HOME/README.md
COPY ./requirements.txt $APP_HOME/requirements.txt

RUN pip install --upgrade setuptools
RUN ls -l $APP_HOME
RUN pip install -r $APP_HOME/requirements.txt
# RUN pwd
# RUN cat $APP_HOME/setup.py

RUN pip install $APP_HOME

WORKDIR $APP_HOME/monailabel-app

