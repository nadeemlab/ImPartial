package org.nadeemlab.impartial;

import okhttp3.*;
import org.json.JSONException;
import org.json.JSONObject;

import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;
import java.io.IOException;
import java.net.URL;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.util.concurrent.TimeUnit;

public class BaseApiClient {
    protected final OkHttpClient httpClient;
    protected URL url;
    protected RequestBody emptyBody = RequestBody.create(null, new byte[0]);

    public BaseApiClient(URL url) {
        this.url = url;
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

    protected void raiseForStatus(Response res) throws IOException {
        if (!res.isSuccessful()) {
            String message = res.message();
            String description = "";
            if (res.body() != null && res.body().contentLength() > 0) {
                try {
                    JSONObject jsonRes = new JSONObject(res.body().string());
                    if (jsonRes.has("message"))
                        message = jsonRes.getString("message");
                    description = jsonRes.has("description") ?
                            jsonRes.getString("description") : jsonRes.getString("detail");
                } catch (JSONException ignore) {}
            }

            throw new IOException(
                    String.format("%d %s \n%s", res.code(), message, description
                    ));
        }
    }

    protected HttpUrl.Builder getHttpUrlBuilder() {
        HttpUrl httpUrl = HttpUrl.parse(url.toString());
        return httpUrl.newBuilder();
    }

    protected Request.Builder getRequestBuilder() {
        return new Request.Builder();
    }
}