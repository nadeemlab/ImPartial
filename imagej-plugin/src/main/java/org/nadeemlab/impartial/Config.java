package org.nadeemlab.impartial;

import java.net.MalformedURLException;
import java.net.URL;

public class Config {
    public static final URL API_URL;

    static {
        try {
            API_URL = new URL("https://impartial.mskcc.org:443");
        } catch (MalformedURLException e) {
            throw new RuntimeException(e);
        }
    }
}
