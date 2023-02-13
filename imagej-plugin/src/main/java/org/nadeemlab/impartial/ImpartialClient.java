package org.nadeemlab.impartial;

import okhttp3.*;
import org.json.JSONObject;

import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;
import java.io.IOException;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.util.concurrent.TimeUnit;

public class ImpartialClient {
    private final OkHttpClient httpClient;
    private final String host = "impartial.mskcc.org";
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
                return new java.security.cert.X509Certificate[]{};
            }
        };

        SSLContext sslContext;
        try {
            sslContext = SSLContext.getInstance("SSL");
            sslContext.init(null, new TrustManager[]{TRUST_ALL_CERTS}, new java.security.SecureRandom());
        } catch (NoSuchAlgorithmException | KeyManagementException e) {
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

    private void raiseForStatus(Response res) throws IOException {
        if (!res.isSuccessful()) {
            JSONObject jsonRes = new JSONObject(res.body().string());
            throw new IOException(
                    String.format("%d %s: %s",
                            res.code(),
                            jsonRes.getString("name"),
                            jsonRes.getString("description"))
            );
        }
    }

    private HttpUrl.Builder getBuilder() {
        return new HttpUrl.Builder()
                .scheme("https")
                .host(host)
                .port(port);
    }

    public String getHost() {
        return host;
    }

    public Integer getPort() {
        return port;
    }

    public JSONObject createSession() throws IOException {
        HttpUrl url = getBuilder()
                .addPathSegments("session/")
                .build();

        RequestBody body = RequestBody.create(null, new byte[0]);

        Request request = new Request.Builder()
                .url(url)
                .put(body)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);

            return new JSONObject(response.body().string());
        }
    }

    public JSONObject sessionStatus(String token) throws IOException {
        HttpUrl url = getBuilder()
                .addPathSegments("session/")
                .build();

        Request request = new Request.Builder()
                .url(url)
                .header("Authorization", token)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);

            return new JSONObject(response.body().string());
        }
    }

    public void stopSession(String token) throws IOException {
        HttpUrl url = getBuilder()
                .addPathSegments("session/")
                .build();

        Request request = new Request.Builder()
                .url(url)
                .header("Authorization", token)
                .delete()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
        }
    }
}