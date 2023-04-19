FROM projectmonai/monailabel:latest

ENV APP_HOME /opt/monai

COPY impartial/ $APP_HOME/impartial/
COPY general/ $APP_HOME/general/
COPY dataprocessing/ $APP_HOME/dataprocessing/

COPY ./requirements.txt $APP_HOME/requirements.txt
RUN pip install -r $APP_HOME/requirements.txt

WORKDIR $APP_HOME/impartial
