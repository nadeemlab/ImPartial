package org.nadeemlab.impartial;

import javax.swing.*;
import java.awt.image.BufferedImage;

public class ImagePanel extends JPanel {
    private ImpartialContentPane controller;
    private MonaiLabelClient monaiClient;
    private JLabel image;
    private JTextField monaiUrl;

    //    private String defaultUrl = "http://10.0.3.117:8000";
    private String defaultUrl = "http://localhost:8000";

    ImagePanel(ImpartialContentPane controller, MonaiLabelClient monaiClient) {
        this.controller = controller;
        this.monaiClient = monaiClient;

        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("image"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5))
        );

        image = new JLabel();

        add(image);
    }

    public void setImage(BufferedImage img) {
        image.setIcon(new ImageIcon(img));
    }

//    class ConnectListener implements ActionListener {
//        public void actionPerformed(ActionEvent e) {
//            URL url;
//            try {
//                url = new URL(monaiUrl.getText());
//            } catch (MalformedURLException ex) {
//                throw new RuntimeException(ex);
//            }
//            monaiClient.setUrl(url);
//            try {
//                monaiClient.getInfo();
//                controller.connect();
//                statusLabel.setText("status: connected");
//            } catch (IllegalArgumentException ignored) {
//                statusLabel.setText("status: disconnected");
//            }
//        }
//    }

}
