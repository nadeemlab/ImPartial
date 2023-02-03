package org.nadeemlab.impartial;

import okhttp3.*;
import org.json.JSONObject;

import javax.net.ssl.*;
import java.io.IOException;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.util.concurrent.TimeUnit;

public class ImpartialClient {
    private final OkHttpClient httpClient;
    private final String host = "internal-alb-e06a1e5-1248497720.us-east-1.elb.amazonaws.com";
    private final Integer port = 443;

    public ImpartialClient() {
        X509TrustManager TRUST_ALL_CERTS = new X509TrustManager() {
            @Override
            public void checkClientTrusted(java.security.cert.X509Certificate[] chain, String authType) {
            }

            @Override
            public void checkServerTrusted(java.security.cert.X509Certificate[] chain, String authType) {
            }

            @Override
            public java.security.cert.X509Certificate[] getAcceptedIssuers() {
                return new java.security.cert.X509Certificate[] {};
            }
        };

        SSLContext sslContext;
        try {
            sslContext = SSLContext.getInstance("SSL");
            sslContext.init(null, new TrustManager[] { TRUST_ALL_CERTS }, new java.security.SecureRandom());
        } catch (NoSuchAlgorithmException|KeyManagementException e) {
            throw new RuntimeException(e);
        }

        OkHttpClient.Builder builder = new OkHttpClient.Builder();

        builder.sslSocketFactory(sslContext.getSocketFactory(), TRUST_ALL_CERTS);
        builder.hostnameVerifier((hostname, session) -> true);

        httpClient = builder
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
                .scheme("https")
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
                .scheme("https")
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

    public JSONObject stopSession(String token) throws IOException {
        HttpUrl url = new HttpUrl.Builder()
                .scheme("https")
                .host(host)
                .port(port)
                .addPathSegments("session/")
                .build();

        Request request = new Request.Builder()
                .url(url)
                .header("Authorization", token)
                .delete()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            return new JSONObject(response.body().string());
        }
    }
}