package org.nadeemlab.impartial;

import okhttp3.*;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.net.URL;

public class MonaiLabelClient extends BaseApiClient {
    public MonaiLabelClient(URL url) {
        super(url);
    }

    public JSONObject getInfo() throws IOException {
        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("info/")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
            return new JSONObject(response.body().string());
        }
    }

    public byte[] getModel(String model) throws IOException {
        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("model/" + model)
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
            return response.body().bytes();
        }
    }

    public void putModel(String model, File file) throws IOException {
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", file.getName(), RequestBody.create(MediaType.parse("application/octet-stream"), file))
                .build();

        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("model/" + model)
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .addHeader("accept", "application/json")
                .addHeader("Content-Type", "multipart/form-data")
                .put(requestBody)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
        }
    }

    public JSONObject postInferJson(String model, String imageId, JSONObject params) throws IOException {
        RequestBody body = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("params", params.toString())
                .build();

        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("infer/" + model)
                .addQueryParameter("image", imageId)
                // .addQueryParameter("output", "json")
                .addQueryParameter("output", "image")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .post(body)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
            return new JSONObject(response.body().string());
        }
    }


    public JSONObject postBatchInferJson(String model, JSONObject params) throws IOException {
        final MediaType JSON = MediaType.parse("application/json; charset=utf-8");

        RequestBody body = RequestBody.create(params.toString(), JSON);

        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("batch/infer/" + model)
                // .addQueryParameter("images", "all")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .post(body)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
            return new JSONObject(response.body().string());
        }
    }

    public JSONObject getBatchInfer() throws IOException {
        HttpUrl.Builder builder = getHttpUrlBuilder();
        builder.addPathSegments("batch/infer");

        builder.addQueryParameter("check_if_running", "false");

        HttpUrl url = builder.build();

        Request request = getRequestBuilder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
            return new JSONObject(response.body().string());
        }
    }

    public JSONObject getTrain(boolean checkIfRunning) throws IOException {
        HttpUrl.Builder builder = getHttpUrlBuilder();
        builder.addPathSegments("train/");

        if (checkIfRunning)
            builder.addQueryParameter("check_if_running", "true");

        HttpUrl url = builder.build();

        Request request = getRequestBuilder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
            return new JSONObject(response.body().string());
        }
    }

    public String postTrain(String model, JSONObject params) throws IOException {
        final MediaType JSON = MediaType.parse("application/json; charset=utf-8");

        RequestBody body = RequestBody.create(params.toString(), JSON);

        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("train/" + model)
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .post(body)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);

            return response.body().string();
        }
    }

    public void deleteTrain() throws IOException {
        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("train/")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .delete()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
        }
    }

    public JSONObject putDatastoreLabel(String imageId, String labelPath) throws IOException {

        final MediaType MEDIA_TYPE_ZIP = MediaType.parse("application/zip");

        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("label", imageId + ".zip",
                        RequestBody.create(MEDIA_TYPE_ZIP, new File(labelPath)))
                .build();

        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("datastore/label")
                .addQueryParameter("image", imageId)
                .addQueryParameter("tag", "final")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .put(requestBody)
                .build();

        try (Response response = this.httpClient.newCall(request).execute()) {
            raiseForStatus(response);

            return new JSONObject(response.body().string());
        }
    }

    public void putDatastoreImage(File imageFile) throws IOException {

        final MediaType MEDIA_TYPE_PNG = MediaType.parse("image/png");

        String imageFileName = imageFile.getName();

        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", imageFileName,
                        RequestBody.create(MEDIA_TYPE_PNG, imageFile))
                .build();

        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("datastore")
                .addPathSegments("image")
                .addQueryParameter("image", imageFileName.substring(0, imageFileName.lastIndexOf(".")))
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .put(requestBody)
                .build();

        try (Response response = this.httpClient.newCall(request).execute()) {
            raiseForStatus(response);
        }
    }

    public void deleteDatastore(String imageId) throws IOException {
        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("datastore")
                .addPathSegments("image")
                .addQueryParameter("id", imageId)
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .delete()
                .build();

        try (Response response = this.httpClient.newCall(request).execute()) {
            raiseForStatus(response);
        }
    }

    public JSONObject getDatastore() throws IOException {
        return getDatastore("all");
    }

    public JSONObject getDatastore(String output) throws IOException {
        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("datastore/")
                .addQueryParameter("output", output)
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
            return new JSONObject(response.body().string());
        }
    }

    public byte[] getDatastoreLabel(String imageId) throws IOException {
        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("datastore/label")
                .addQueryParameter("label", imageId)
                .addQueryParameter("tag", "final")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);

            return response.body().bytes();
        }
    }

//     public byte[] getDatastoreLabel(String imageId, String tag) throws IOException {
//         HttpUrl url = getHttpUrlBuilder()
//                 .addPathSegments("datastore/label")
//                 .addQueryParameter("label", imageId)
//                 .addQueryParameter("tag", tag)
//                 .build();

//         Request request = getRequestBuilder()
//                 .url(url)
//                 .build();

//         try (Response response = httpClient.newCall(request).execute()) {
//             raiseForStatus(response);

//             return response.body().bytes();
//         }
//     }

    public boolean headDatastoreImage(String imageId) throws IOException {
        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("datastore/image")
                .addQueryParameter("image", imageId)
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .head()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            return response.code() == 200;
        }
    }

    public byte[] getDatastoreImage(String imageId) throws IOException {
        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("datastore/image")
                .addQueryParameter("image", imageId)
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);

            return response.body().bytes();
        }
    }

    public String getLogs() throws IOException {
        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("logs")
                .addQueryParameter("text", "true")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);

            return response.body().string();
        }
    }
}