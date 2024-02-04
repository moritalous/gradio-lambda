FROM public.ecr.aws/docker/library/python:3.12.0-slim
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.1 /lambda-adapter /opt/extensions/lambda-adapter

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

WORKDIR /var/task

ENV PORT=8000
ENV AWS_LWA_READINESS_CHECK_PROTOCOL=tcp 
ENV AWS_LWA_ASYNC_INIT=true
ENV AWS_LWA_INVOKE_MODE=response_stream
ENV GRADIO_TEMP_DIR=/tmp/gradio_tmp
ENV GRADIO_EXAMPLES_CACHE=/tmp/gradio_cached_examples
ENV MPLCONFIGDIR=/tmp/matplotlib

COPY *.py ./
CMD python app.py
