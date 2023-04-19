package org.nadeemlab.impartial;

import okhttp3.*;
import org.json.JSONObject;

import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.security.KeyManagementException;
import java.security.NoSuchAlgorithmException;
import java.util.concurrent.TimeUnit;

public class BaseApiClient {
    protected final OkHttpClient httpClient;
    protected String protocol;
    protected String host;
    protected Integer port;
    protected String path = "";
    protected String token;
    protected RequestBody emptyBody = RequestBody.create(null, new byte[0]);

    public BaseApiClient() {
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
            String name = res.message();
            String description = "";
            if (res.body() != null && res.body().contentLength() > 0) {
                JSONObject jsonRes = new JSONObject(res.body().string());
                if (jsonRes.has("name"))
                    name = jsonRes.getString("name");
                description = jsonRes.has("description") ?
                        jsonRes.getString("description") : jsonRes.getString("detail");
            }

            throw new IOException(
                    String.format("%d %s: %s", res.code(), name, description
                    ));
        }
    }

    protected HttpUrl.Builder getHttpUrlBuilder() {
        return new HttpUrl.Builder()
                .scheme(protocol)
                .host(host)
                .port(port)
                .addPathSegments(path);
    }

    protected Request.Builder getRequestBuilder() {
        Request.Builder builder = new Request.Builder();

        if (token != null)
            builder.addHeader("Authorization", token);

        return builder;
    }

    public URL getUrl() {
        try {
            return new URL(String.format("%s://%s:%s/%s", protocol, host, port, path));
        } catch (MalformedURLException ignore) {
            return null;
        }
    }

    public void setUrl(URL url) {
        protocol = url.getProtocol();
        host = url.getHost();
        port = url.getPort() > 1 ? url.getPort() : url.getDefaultPort();
        path = url.getPath();
    }

    public String getProtocol() {
        return protocol;
    }

    public String getHost() {
        return host;
    }

    public Integer getPort() {
        return port;
    }

    public void setToken(String token) {
        this.token = token;
    }

    public boolean hasToken() {
        return this.token != null;
    }

}