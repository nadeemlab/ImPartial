package org.nadeemlab.impartial;

import okhttp3.HttpUrl;
import okhttp3.Request;

import java.net.URL;

public class ProxyMonaiLabelClient extends MonaiLabelClient{
    private final String token;
    public ProxyMonaiLabelClient(URL url, String token) {
        super(url);
        this.token = token;
    }

    protected Request.Builder getRequestBuilder() {
        Request.Builder builder = new Request.Builder();
        builder.addHeader("Authorization", token);

        return builder;
    }

    protected HttpUrl.Builder getHttpUrlBuilder() {
        HttpUrl.Builder builder = super.getHttpUrlBuilder();
        builder.addPathSegments("proxy");

        return builder;
    }
}
