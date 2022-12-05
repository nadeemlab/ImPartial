package org.nadeemlab.impartial;

import okhttp3.*;
import org.json.JSONObject;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class ImpartialClient {
    private final OkHttpClient httpClient;
    private final String host = "internal-alb-a37704c-1767807991.us-east-1.elb.amazonaws.com";
    private final Integer port = 80;

    public ImpartialClient() {
        httpClient = new OkHttpClient.Builder()
                .connectTimeout(120, TimeUnit.SECONDS)
                .writeTimeout(10, TimeUnit.SECONDS)
                .readTimeout(120, TimeUnit.SECONDS)
                .build();
    }

    public String getHost() {
        return host;
    }

    public Integer getPort() {
        return port;
    }

    public JSONObject createSession() throws IOException {
        HttpUrl url = new HttpUrl.Builder()
                .scheme("http")
                .host(host)
                .port(port)
                .addPathSegments("session/")
                .build();

        RequestBody body = RequestBody.create(null, new byte[0]);

        Request request = new Request.Builder()
                .url(url)
                .put(body)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            return new JSONObject(response.body().string());
        }
    }

    public JSONObject sessionStatus(String token) throws IOException {
        HttpUrl url = new HttpUrl.Builder()
                .scheme("http")
                .host(host)
                .port(port)
                .addPathSegments("session/")
                .build();

        Request request = new Request.Builder()
                .url(url)
                .header("Authorization", token)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            return new JSONObject(response.body().string());
        }
    }
}