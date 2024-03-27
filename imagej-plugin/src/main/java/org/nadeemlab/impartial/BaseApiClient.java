package org.nadeemlab.impartial;

import okhttp3.*;
import okio.Buffer;

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
                .connectTimeout(60, TimeUnit.SECONDS)
                .writeTimeout(300, TimeUnit.SECONDS)
                .readTimeout(300, TimeUnit.SECONDS)
                .build();
    }

    protected void raiseForStatus(Response response) throws IOException {
        logResponseInfo(response);

        if (!response.isSuccessful()) {
            String responseBody = "";

            if (response.body() != null && response.body().contentLength() > 0) {
                responseBody = response.body().string();
            }

            String error = getErrorLog(response, responseBody);

            System.out.println(error);

            throw new IOException(
                    String.format("%d %s \n%s", response.code(), "Error message: ", error
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

    protected Response callHttpClientAndLogRequestInfo(Request request) throws IOException {
        System.out.println("=========================<HTTP QUERY INFO>===================================");
        String requestMethod = request.method();
        System.out.println("Request method: " + requestMethod);
        System.out.println("Request url: " + request.url());
        System.out.println("Request headers: " + request.headers());

        if (request.body() != null && request.body().contentLength() > 0) {
            System.out.println("Request body: " + request.body().toString());
            System.out.println("Request body: " + bodyToString(request));
        }

        return httpClient.newCall(request).execute();
    }

    private void logResponseInfo(Response response) {
        int responseCode = response.code();
        System.out.println("Response for url: " + response.request().url());
        System.out.println("Response code: " + responseCode);
        System.out.println("=========================</HTTP QUERY INFO>===================================");
    }

    private String getErrorLog(Response response, String responseBody) throws IOException {
        int responseCode = response.code();
        String requestMethod = response.request().method();

        String requestBody = null;
        if (response.request().body() != null && response.request().body().contentLength() > 0) {
            requestBody = response.request().body().toString();
        }

        return "++++++++++++++++++++++++++++++++<ERROR LOG>++++++++++++++++++++++++++++++++++++++" + "\n" +
                "Response for url: " + response.request().url() + "\n" +
                "Request method: " + requestMethod + "\n" +
                "Request headers: " + response.request().headers() + "\n" +
                "Request body: " + requestBody + "\n" +
                "Request body: " + bodyToString(response.request()) + "\n" +
                "~~~~~~~~~~~~~~~~~~~~~~~\n" +
                "Response headers: " + response.headers() + "\n" +
                "Response code: " + responseCode + "\n" +
                "Response body: " + responseBody + "\n" +
                "++++++++++++++++++++++++++++++++</ERROR LOG>++++++++++++++++++++++++++++++++++++++" + "\n";
    }

    private static String bodyToString(final Request request) {
        try (Buffer buffer = new Buffer()) {
            final Request copy = request.newBuilder().build();
            copy.body().writeTo(buffer);
            return buffer.readUtf8();
        } catch (final IOException e) {
            return "Body could not be parsed";
        }
    }
}