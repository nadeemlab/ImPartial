package org.nadeemlab.impartial;

import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class UserSession {
    private String id;
    private String date;
    private int numImages;
    private int numLabels;

    public UserSession(String id, String date, int numImages, int numLabels) {
        this.id = id;
        this.date = date;
        this.numImages = numImages;
        this.numLabels = numLabels;
    }

    public String getId() {
        return id;
    }

    public String getDate() {
        return date;
    }

    public int getNumImages() {
        return numImages;
    }

    public int getNumLabels() {
        return numLabels;
    }

    public void setId(String id) {
        this.id = id;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public void setNumImages(int numImages) {
        this.numImages = numImages;
    }

    public void setNumLabels(int numLabels) {
        this.numLabels = numLabels;
    }

    @Override
    public String toString() {
        return "Session{" +
                "id=" + id +
                ", date='" + date + '\'' +
                ", numImages=" + numImages +
                ", numLabels=" + numLabels +
                '}';
    }

    public String getParsedDate() {
        return parseDate(date);
    }

    private String parseDate(String inputDate) {
        DateFormat input = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSSSS");
        DateFormat output = new SimpleDateFormat("MMM d, h:mma");

        Date date = null;
        try {
            date = input.parse(inputDate);
        } catch (ParseException ignore) {
        }

        return output.format(date);
    }
}
