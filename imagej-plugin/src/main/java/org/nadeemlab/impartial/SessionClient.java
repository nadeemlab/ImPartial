package org.nadeemlab.impartial;

import okhttp3.FormBody;
import okhttp3.HttpUrl;
import okhttp3.Request;
import okhttp3.Response;
import org.json.JSONObject;

import java.io.IOException;
import java.net.URL;

public class SessionClient extends BaseApiClient {
    private String token;

    public SessionClient(URL url) {
        super(url);
    }

    protected Request.Builder getRequestBuilder() {
        Request.Builder builder = new Request.Builder();
        builder.addHeader("Authorization", token);

        return builder;
    }

    public JSONObject getSessions() throws IOException {
        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("session")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
            return new JSONObject(response.body().string());
        }
    }

    public JSONObject getSessionDetails() throws IOException {
        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("session")
                .addPathSegments("details")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
            return new JSONObject(response.body().string());
        }
    }

    public void stopSession() throws IOException {
        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("session")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .delete()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
        }
    }

    public void restoreSession(String sessionId) throws IOException {
        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("session")
                .addPathSegments("restore")
                .addQueryParameter("session_id", sessionId)
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .post(emptyBody)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
        }
    }

    public String postSession() throws IOException {
        HttpUrl.Builder httpUrlBuilder = getHttpUrlBuilder();
        httpUrlBuilder.addPathSegments("session");

        HttpUrl url = httpUrlBuilder.build();

        Request request = getRequestBuilder()
                .url(url)
                .post(emptyBody)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
            return new JSONObject(response.body().string()).getString("session_id");
        }
    }

    public void extendSession() throws IOException {
        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("session")
                .addPathSegments("extend")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .post(emptyBody)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
        }
    }

    public String postLogin(String username, String password) throws IOException {
        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("login")
                .build();

        Request request = new Request.Builder()
                .url(url)
                .post(new FormBody.Builder()
                        .add("username", username)
                        .add("password", password)
                        .build())
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
            return new JSONObject(response.body().string()).getString("token");
        }
    }

    public void setToken(String token) {
        this.token = token;
    }
}