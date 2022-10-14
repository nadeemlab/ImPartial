package org.nadeemlab.impartial;

import okhttp3.*;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.concurrent.TimeUnit;

public class MonaiLabelClient {
    private final OkHttpClient httpClient;
    private String host;
    private Integer port;

    public MonaiLabelClient() {
        httpClient = new OkHttpClient.Builder()
                .connectTimeout(120, TimeUnit.SECONDS)
                .writeTimeout(10, TimeUnit.SECONDS)
                .readTimeout(120, TimeUnit.SECONDS)
                .build();
    }

    public void setUrl(URL url) {
        this.host = url.getHost();
        this.port = url.getPort();
    }

    public JSONObject getInfo() {
        HttpUrl url = new HttpUrl.Builder()
                .scheme("http")
                .host(host)
                .port(port)
                .addPathSegments("info")
                .build();

        Request request = new Request.Builder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            return new JSONObject(response.body().string());
        } catch (Exception e) {
            return new JSONObject();
        }
    }

    public JSONObject postInferJson(String model, String imageId, JSONObject params) throws IOException {
        RequestBody body = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("params", params.toString())
                .build();

        HttpUrl url = new HttpUrl.Builder()
                .scheme("http")
                .host(host)
                .port(port)
                .addPathSegments("infer/" + model)
                .addQueryParameter("image", imageId)
                .addQueryParameter("output", "json")
                .build();

        Request request = new Request.Builder()
                .url(url)
                .post(body)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            return new JSONObject(response.body().string());
        }
    }

    public byte[] postInferBytes(String model, String imageId, JSONObject params) {
        RequestBody body = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("params", params.toString())
                .build();

        HttpUrl url = new HttpUrl.Builder()
                .scheme("http")
                .host(host)
                .port(port)
                .addPathSegments("infer/" + model)
                .addQueryParameter("image", imageId)
                .addQueryParameter("output", "image")
                .build();

        Request request = new Request.Builder()
                .url(url)
                .post(body)
                .build();
        try (Response response = httpClient.newCall(request).execute()) {
            return response.body().bytes();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public JSONObject getTrain() {
        HttpUrl url = new HttpUrl.Builder()
                .scheme("http")
                .host(host)
                .port(port)
                .addPathSegments("train/")
                .build();

        Request request = new Request.Builder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            return new JSONObject(response.body().string());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public String postTrain(String model, JSONObject params) {
        final MediaType JSON = MediaType.parse("application/json; charset=utf-8");

        RequestBody body = RequestBody.create(params.toString(), JSON);

        HttpUrl url = new HttpUrl.Builder()
                .scheme("http")
                .host(host)
                .port(port)
                .addPathSegments("train/" + model)
                .build();

        Request request = new Request.Builder()
                .url(url)
                .post(body)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            return response.body().string();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public String deleteTrain() {
        HttpUrl url = new HttpUrl.Builder()
                .scheme("http")
                .host(host)
                .port(port)
                .addPathSegments("train/")
                .build();

        Request request = new Request.Builder()
                .url(url)
                .delete()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            return response.body().string();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public JSONObject postActiveLearning(String strategy) {
//		TODO: consider the case when all images in the dataset are
//		already labeled and this endpoint returns an empty response
        HttpUrl url = new HttpUrl.Builder()
                .scheme("http")
                .host(host)
                .port(port)
                .addPathSegments("activelearning/" + strategy)
                .build();

        Request request = new Request.Builder()
                .url(url)
                .post(RequestBody.create(new byte[0]))
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            return new JSONObject(response.body().string());
        } catch (Exception e) {
            return new JSONObject();
        }
    }

    public JSONObject putDatastoreLabel(String imageId, String labelPath) {

        final MediaType MEDIA_TYPE_ZIP = MediaType.parse("application/zip");

        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("label", imageId + ".zip",
                        RequestBody.create(MEDIA_TYPE_ZIP, new File(labelPath)))
                .build();

        HttpUrl url = new HttpUrl.Builder()
                .scheme("http")
                .host(host)
                .port(port)
                .addPathSegments("datastore/label")
                .addQueryParameter("image", imageId)
                .addQueryParameter("tag", "final")
                .build();

        Request request = new Request.Builder()
                .url(url)
                .put(requestBody)
                .build();

        try (Response response = this.httpClient.newCall(request).execute()) {
            return new JSONObject(response.body().string());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public JSONObject putDatastore(File imageFile) {

        final MediaType MEDIA_TYPE_PNG = MediaType.parse("image/png");

        String imageFileName = imageFile.getName();

        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("file", imageFileName,
                        RequestBody.create(MEDIA_TYPE_PNG, imageFile))
                .build();

        HttpUrl url = new HttpUrl.Builder()
                .scheme("http")
                .host(host)
                .port(port)
                .addPathSegments("datastore")
                .addQueryParameter("image", imageFileName.substring(0, imageFileName.lastIndexOf(".")))
                .build();

        Request request = new Request.Builder()
                .url(url)
                .put(requestBody)
                .build();

        try (Response response = this.httpClient.newCall(request).execute()) {
            return new JSONObject(response.body().string());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void deleteDatastore(String imageId) throws IOException {
        HttpUrl url = new HttpUrl.Builder()
                .scheme("http")
                .host(host)
                .port(port)
                .addPathSegments("datastore")
                .addQueryParameter("id", imageId)
                .build();

        Request request = new Request.Builder()
                .url(url)
                .delete()
                .build();

        httpClient.newCall(request).execute().close();
    }

    public JSONObject getDatastore() throws IOException {
        HttpUrl url = new HttpUrl.Builder()
                .scheme("http")
                .host(this.host)
                .port(port)
                .addPathSegments("datastore")
                .addQueryParameter("output", "all")
                .build();

        Request request = new Request.Builder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            return new JSONObject(response.body().string());
        }
    }

    public byte[] getDatastoreLabel(String imageId) throws IOException {
        HttpUrl url = new HttpUrl.Builder()
                .scheme("http")
                .host(this.host)
                .port(port)
                .addPathSegments("datastore/label")
                .addQueryParameter("label", imageId)
                .addQueryParameter("tag", "final")
                .build();

        Request request = new Request.Builder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            return response.body().bytes();
        }
    }

    public byte[] getDatastoreImage(String imageId) {
        HttpUrl url = new HttpUrl.Builder()
                .scheme("http")
                .host(this.host)
                .port(port)
                .addPathSegments("datastore/image")
                .addQueryParameter("image", imageId)
                .build();

        Request request = new Request.Builder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            return response.body().bytes();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public JSONObject downloadImage(String image) {
        HttpUrl url = new HttpUrl.Builder()
                .scheme("http")
                .host(this.host)
                .port(port)
                .addPathSegments("datastore/image")
                .addQueryParameter("image", image)
                .build();

        Request request = new Request.Builder()
                .url(url)
                .build();

        try (Response response = this.httpClient.newCall(request).execute()) {
            return new JSONObject(response.body().string());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}