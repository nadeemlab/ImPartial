package org.nadeemlab.impartial;

import okhttp3.*;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.concurrent.TimeUnit;

public class MonaiLabelClient {
    private final OkHttpClient httpClient;
    private URL url;
    private String token;

    public MonaiLabelClient() {
        httpClient = new OkHttpClient.Builder()
                .connectTimeout(120, TimeUnit.SECONDS)
                .writeTimeout(10, TimeUnit.SECONDS)
                .readTimeout(120, TimeUnit.SECONDS)
                .build();
        try {
            url = new URL("http", "localhost", 8000, "");
        } catch (MalformedURLException ignore) {}
    }

    public void setUrl(URL url) {
        this.url = url;
    }

    public void setToken(String token) {
        this.token = token;
    }

    private HttpUrl.Builder getHttpBuilder() {
        HttpUrl.Builder builder =  new HttpUrl.Builder()
                .scheme(url.getProtocol())
                .host(url.getHost())
                .port(url.getPort() > 1 ? url.getPort() : 80);

        if (token != null)
            builder.addPathSegments("proxy");

        return builder;
    }

    private Request.Builder getRequestBuilder() {
        Request.Builder builder = new Request.Builder();

        if (token != null)
            builder.addHeader("Authorization", token);

        return builder;
    }

    public JSONObject getInfo() throws IOException {
        HttpUrl url = getHttpBuilder()
                .addPathSegments("info")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            return new JSONObject(response.body().string());
        }
    }

    public JSONObject postInferJson(String model, String imageId, JSONObject params) throws IOException {
        RequestBody body = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("params", params.toString())
                .build();

        HttpUrl url = getHttpBuilder()
                .addPathSegments("infer/" + model)
                .addQueryParameter("image", imageId)
                .addQueryParameter("output", "json")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .post(body)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            return new JSONObject(response.body().string());
        }
    }

    public byte[] postInferBytes(String model, String imageId, JSONObject params) throws IOException {
        RequestBody body = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("params", params.toString())
                .build();

        HttpUrl url = getHttpBuilder()
                .addPathSegments("infer/" + model)
                .addQueryParameter("image", imageId)
                .addQueryParameter("output", "image")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .post(body)
                .build();
        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            return response.body().bytes();
        }
    }

    public JSONObject getTrain() throws IOException {
        HttpUrl url = getHttpBuilder()
                .addPathSegments("train/")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            return new JSONObject(response.body().string());
        }
    }

    public String postTrain(String model, JSONObject params) throws IOException {
        final MediaType JSON = MediaType.parse("application/json; charset=utf-8");

        RequestBody body = RequestBody.create(params.toString(), JSON);

        HttpUrl url = getHttpBuilder()
                .addPathSegments("train/" + model)
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .post(body)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            return response.body().string();
        }
    }

    public String deleteTrain() throws IOException {
        HttpUrl url = getHttpBuilder()
                .addPathSegments("train/")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .delete()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            return response.body().string();
        }
    }

    public JSONObject postActiveLearning(String strategy) throws IOException {
//		TODO: consider the case when all images in the dataset are
//		already labeled and this endpoint returns an empty response
        HttpUrl url = getHttpBuilder()
                .addPathSegments("activelearning/" + strategy)
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .post(RequestBody.create(new byte[0]))
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            return new JSONObject(response.body().string());
        }
    }

    public JSONObject putDatastoreLabel(String imageId, String labelPath) throws IOException {

        final MediaType MEDIA_TYPE_ZIP = MediaType.parse("application/zip");

        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("label", imageId + ".zip",
                        RequestBody.create(MEDIA_TYPE_ZIP, new File(labelPath)))
                .build();

        HttpUrl url = getHttpBuilder()
                .addPathSegments("datastore/label")
                .addQueryParameter("image", imageId)
                .addQueryParameter("tag", "final")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .put(requestBody)
                .build();

        try (Response response = this.httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            return new JSONObject(response.body().string());
        }
    }

    public JSONObject putDatastore(File imageFile) throws IOException {

        final MediaType MEDIA_TYPE_PNG = MediaType.parse("image/png");

        String imageFileName = imageFile.getName();

        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", imageFileName,
                        RequestBody.create(MEDIA_TYPE_PNG, imageFile))
                .build();

        HttpUrl url = getHttpBuilder()
                .addPathSegments("datastore")
                .addQueryParameter("image", imageFileName.substring(0, imageFileName.lastIndexOf(".")))
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .put(requestBody)
                .build();

        try (Response response = this.httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            return new JSONObject(response.body().string());
        }
    }

    public void deleteDatastore(String imageId) throws IOException {
        HttpUrl url = getHttpBuilder()
                .addPathSegments("datastore")
                .addQueryParameter("id", imageId)
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .delete()
                .build();

        httpClient.newCall(request).execute().close();
    }

    public JSONObject getDatastore() throws IOException {
        HttpUrl url = getHttpBuilder()
                .addPathSegments("datastore")
                .addQueryParameter("output", "all")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            return new JSONObject(response.body().string());
        }
    }

    public byte[] getDatastoreLabel(String imageId) throws IOException {
        HttpUrl url = getHttpBuilder()
                .addPathSegments("datastore/label")
                .addQueryParameter("label", imageId)
                .addQueryParameter("tag", "final")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            return response.body().bytes();
        }
    }

    public byte[] getDatastoreImage(String imageId) throws IOException {
        HttpUrl url = getHttpBuilder()
                .addPathSegments("datastore/image")
                .addQueryParameter("image", imageId)
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            return response.body().bytes();
        }
    }

    public JSONObject downloadImage(String image) throws IOException {
        HttpUrl url = getHttpBuilder()
                .addPathSegments("datastore/image")
                .addQueryParameter("image", image)
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .build();

        try (Response response = this.httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            return new JSONObject(response.body().string());
        }
    }
}