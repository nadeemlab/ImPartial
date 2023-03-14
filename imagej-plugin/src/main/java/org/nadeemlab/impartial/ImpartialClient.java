package org.nadeemlab.impartial;

import okhttp3.HttpUrl;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import org.json.JSONObject;

import java.io.IOException;

public class ImpartialClient extends BaseApiClient {
    public ImpartialClient() {
        super();

        protocol = "https";
         host = "impartial.mskcc.org";
//        host = "internal-alb-e06a1e5-1248497720.us-east-1.elb.amazonaws.com";
        port = 443;
    }

    public JSONObject createSession() throws IOException {
        HttpUrl url = getHttpUrlBuilder()
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

    public JSONObject sessionStatus() throws IOException {
        HttpUrl url = getHttpUrlBuilder()
                .addPathSegments("session/")
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
                .addPathSegments("session/")
                .build();

        Request request = getRequestBuilder()
                .url(url)
                .delete()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            raiseForStatus(response);
        }
    }
}