package org.nadeemlab.impartial;

import okhttp3.HttpUrl;
import okhttp3.Request;
import okhttp3.Response;
import org.json.JSONObject;

import java.io.IOException;

public class SessionClient extends BaseApiClient {
    public SessionClient() {
        super();

       protocol = "https";
       host = "impartial.mskcc.org";
       port = 443;
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

    public JSONObject restoreSession(String sessionId) throws IOException {
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
            return new JSONObject(response.body().string());
        }
    }

    public JSONObject postSession(String sessionId) throws IOException {
        HttpUrl.Builder httpUrlBuilder = getHttpUrlBuilder();
        httpUrlBuilder.addPathSegments("session");
        httpUrlBuilder.addQueryParameter("session_id", sessionId);

        HttpUrl url = httpUrlBuilder.build();

        Request request = new Request.Builder()
                .url(url)
                .post(emptyBody)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
            return new JSONObject(response.body().string());
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

}