FROM public.ecr.aws/docker/library/python:3.9

WORKDIR /src

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    libssl-dev


COPY requirements.txt /src/
RUN pip install -r requirements.txt

COPY . /src/

EXPOSE 8000

# Use CMD over ENTRYPOINT because it allows overrides
CMD ["./docker-entrypoint.sh"]