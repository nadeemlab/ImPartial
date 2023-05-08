package org.nadeemlab.impartial;

import org.json.JSONObject;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;

public class ServerPanel extends JPanel {
    private final ImpartialController controller;
    private final JTextField monaiUrlTextField;
    private final JPanel userPanel;
    private JLabel sessionLabel;
    private JLabel usernameLabel;
    private JCheckBox requestServerCheckBox;
    private JButton startStopButton;
    private String url = "http://localhost:8000";
    private JButton selectSessionButton;

    ServerPanel(ImpartialController controller) {
        this.controller = controller;

        setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createTitledBorder("SERVER"),
                BorderFactory.createEmptyBorder(5, 5, 5, 5))
        );

        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setAlignmentX(LEFT_ALIGNMENT);

        monaiUrlTextField = new JTextField(url, 16);
        monaiUrlTextField.setAlignmentX(LEFT_ALIGNMENT);
        monaiUrlTextField.setText(url);

        userPanel = createUserPanel();
        userPanel.setVisible(false);

        add(monaiUrlTextField);
        add(createButtonsPanel());
        add(userPanel);

        monaiUrlTextField.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                startStopButton.setEnabled(true);
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                startStopButton.setEnabled(true);
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                startStopButton.setEnabled(true);
            }
        });
    }

    private JPanel createUserPanel() {
        JPanel userPanel = new JPanel();
        userPanel.setLayout(new BoxLayout(userPanel, BoxLayout.Y_AXIS));
        userPanel.setAlignmentX(LEFT_ALIGNMENT);

        usernameLabel = new JLabel();
        userPanel.setToolTipText("username of the currently logged-in user");

        userPanel.add(usernameLabel);
        userPanel.add(createSessionPanel());

        return userPanel;
    }

    private JPanel createButtonsPanel() {
        JPanel panel = new JPanel();
        panel.setAlignmentX(LEFT_ALIGNMENT);
        panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));

        startStopButton = new JButton("start");

        panel.add(createRequestServerCheckbox());
        panel.add(Box.createHorizontalGlue());
        panel.add(startStopButton);

        panel.add(Box.createVerticalGlue());

        startStopButton.addActionListener(e -> {
            selectSessionButton.setEnabled(false);
            if (startStopButton.getText().equals("start")) {
                if (!requestServerCheckBox.isSelected()) {
                    url = monaiUrlTextField.getText();
                    controller.start();
                }
                else {
                    LoginDialog loginDialog = new LoginDialog(controller);
                    loginDialog.setVisible(true);

                    if (loginDialog.isLoginSuccessful()) {
                        usernameLabel.setText(
                                String.format("<html> <strong>user</strong> %s</html>", loginDialog.getUsername())
                        );
                        startStopButton.setEnabled(false);
                        controller.start();
                    }
                }
            } else {
                controller.stop();
            }
        });

        return panel;
    }

    private JCheckBox createRequestServerCheckbox() {
        requestServerCheckBox = new JCheckBox("request server");

        requestServerCheckBox.addActionListener(e -> {
            if (requestServerCheckBox.isSelected()) {
                monaiUrlTextField.setEnabled(false);
                monaiUrlTextField.setText("https://impartial.mskcc.org");
            } else {
                monaiUrlTextField.setEnabled(true);
                monaiUrlTextField.setText(url);
            }
        });

        return requestServerCheckBox;
    }

    private JPanel createSessionPanel() {
        JPanel panel = new JPanel();
        panel.setAlignmentX(LEFT_ALIGNMENT);
        panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));

        sessionLabel = new JLabel("<html> <strong>session</strong>");

        selectSessionButton = new JButton("select");
        selectSessionButton.addActionListener(e -> {
            JSONObject sessions = controller.getSessions();

            ArrayList<UserSession> userSessions = new ArrayList<>();

            if (sessions.length() > 0) {
                String[] keys = JSONObject.getNames(sessions);
                for (String session_id : keys) {
                    JSONObject session = (JSONObject) sessions.get(session_id);

                    userSessions.add(new UserSession(
                            session_id,
                            session.getString("created_at"),
                            session.getJSONArray("images").length(),
                            session.getJSONArray("labels").length()
                    ));
                }
            }

            UserSessionDialog dialog = new UserSessionDialog(controller.getFrame(), userSessions);
            dialog.setVisible(true);
            if (dialog.getSelectedSession() != null) {
                controller.setSession(dialog.getSelectedSession());
            }
        });

        panel.add(sessionLabel);
        panel.add(Box.createHorizontalGlue());
        panel.add(selectSessionButton);

        return panel;
    }

    public void setSession(String sessionId) {
        sessionLabel.setText(String.format(
                "<html> <strong>session</strong> %s </html>",
                sessionId.substring(0, Math.min(sessionId.length(), 8))
        ));
    }

    public URL getUrl() throws MalformedURLException {
        return new URL(url);
    }

    public boolean getRequestServerCheckBox() {
        return requestServerCheckBox.isSelected();
    }

    public void onStarted() {
        startStopButton.setText("stop");
        startStopButton.setEnabled(true);
        selectSessionButton.setEnabled(true);
        requestServerCheckBox.setEnabled(false);
        if (requestServerCheckBox.isSelected()) {
            userPanel.setVisible(true);
            controller.getFrame().pack();
        }
    }

    public void onStopped() {
        startStopButton.setText("start");
        startStopButton.setEnabled(true);
        selectSessionButton.setEnabled(true);
        requestServerCheckBox.setEnabled(true);
        sessionLabel.setText("<html> <strong>session</strong> </html>");
        userPanel.setVisible(false);
        controller.getFrame().pack();
    }
}

