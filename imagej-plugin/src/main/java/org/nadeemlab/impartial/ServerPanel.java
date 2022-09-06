package org.nadeemlab.impartial;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.net.MalformedURLException;
import java.net.URL;

public class ServerPanel extends JPanel {
    private ImpartialController controller;
    private JLabel statusLabel;
    private JTextField monaiUrl;

//    private String defaultUrl = "http://10.0.3.117:8000";
    private String defaultUrl = "http://localhost:8000";

    ServerPanel(ImpartialController controller) {
        this.controller = controller;

        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("server"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5))
        );

        JPanel content = new JPanel();
        content.setLayout(new BoxLayout(content, BoxLayout.PAGE_AXIS));

        monaiUrl = new JTextField(defaultUrl, 16);
        monaiUrl.setAlignmentX(LEFT_ALIGNMENT);

        JPanel innerPanel = new JPanel();
        innerPanel.setLayout(new BoxLayout(innerPanel, BoxLayout.LINE_AXIS));
        innerPanel.setAlignmentX(Component.LEFT_ALIGNMENT);

        JButton connectButton = new JButton("connect");
        connectButton.addActionListener(e -> {
            URL url;
            try {
                url = new URL(monaiUrl.getText());
            } catch (MalformedURLException ex) {
                throw new RuntimeException(ex);
            }
            controller.setMonaiClientUrl(url);

            try {
                controller.connect();
                statusLabel.setText("status: connected");
            } catch (IllegalArgumentException ignored) {
                statusLabel.setText("status: disconnected");
            }
        });

        statusLabel = new JLabel();
        statusLabel.setText("status: disconnected");
//        statusLabel.setIcon(new ImageIcon("check-circle.png"));

        innerPanel.add(connectButton);
        innerPanel.add(statusLabel);
        innerPanel.add(Box.createHorizontalGlue());

        content.add(monaiUrl);
        content.add(innerPanel);
//        content.add(statusLabel);

        add(content);
    }

}
