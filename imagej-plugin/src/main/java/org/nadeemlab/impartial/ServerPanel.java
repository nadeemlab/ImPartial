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

        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setAlignmentX(LEFT_ALIGNMENT);
        setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));

        JLabel title = new JLabel("Server");
        title.setFont(new Font("sans-serif", Font.PLAIN, 15));
        add(title);
        add(Box.createVerticalStrut(10));

        monaiUrlTextField = new JTextField(url, 12);
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
        JPanel userPanel = new JPanel(new GridBagLayout());
        userPanel.setAlignmentX(LEFT_ALIGNMENT);

        usernameLabel = new JLabel();
        userPanel.setToolTipText("Username of the currently logged-in user");

        sessionLabel = new JLabel("Session");

        selectSessionButton = new JButton("Select");
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

        addRow(userPanel, "User", usernameLabel);
        addRow(userPanel, "Session", sessionLabel, selectSessionButton);

        return userPanel;
    }

    private void addRow(JPanel panel, String labelText, JLabel valueLabel, JButton... buttons) {
        GridBagConstraints c = new GridBagConstraints();
        c.gridx = 0;
        c.gridy = GridBagConstraints.RELATIVE;
        c.anchor = GridBagConstraints.LINE_START;
        c.insets = new Insets(0, 0, 5, 0);

        JLabel label = new JLabel(labelText);
        panel.add(label, c);

        c.gridx = 1;
        c.weightx = 1;
        c.anchor = GridBagConstraints.LINE_START;
        c.insets = new Insets(0, 5, 5, 0);

        panel.add(valueLabel, c);

        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 0, 0));
        for (JButton button : buttons) {
            buttonPanel.add(button);
        }
        c.gridx = 2;
        c.weightx = 0;
        c.insets = new Insets(0, 0, 5, 0);
        panel.add(buttonPanel, c);
    }

    private JPanel createButtonsPanel() {
        JPanel panel = new JPanel();
        panel.setAlignmentX(LEFT_ALIGNMENT);
        panel.setLayout(new BoxLayout(panel, BoxLayout.X_AXIS));
        panel.setBorder(BorderFactory.createEmptyBorder(0, 0, 0, 0));

        startStopButton = new JButton("Start");

        panel.add(createRequestServerCheckbox());
        panel.add(Box.createHorizontalGlue());
        panel.add(startStopButton);

        panel.add(Box.createVerticalGlue());

        startStopButton.addActionListener(e -> {
            selectSessionButton.setEnabled(false);
            if (startStopButton.getText().equals("Start")) {
                if (!requestServerCheckBox.isSelected()) {
                    url = monaiUrlTextField.getText();
                    controller.start();
                }
                else {
                    LoginDialog loginDialog = new LoginDialog(controller);
                    loginDialog.setVisible(true);

                    if (loginDialog.isLoginSuccessful()) {
                        usernameLabel.setText(loginDialog.getUsername());
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
        requestServerCheckBox = new JCheckBox("Request server");

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

        sessionLabel = new JLabel("Session");

        selectSessionButton = new JButton("Select");
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
        sessionLabel.setText(sessionId.substring(0, Math.min(sessionId.length(), 8)));
    }

    public URL getUrl() throws MalformedURLException {
        return new URL(url);
    }

    public boolean getRequestServerCheckBox() {
        return requestServerCheckBox.isSelected();
    }

    public void onStarted() {
        startStopButton.setText("Stop");
        startStopButton.setEnabled(true);
        selectSessionButton.setEnabled(true);
        requestServerCheckBox.setEnabled(false);
        if (requestServerCheckBox.isSelected()) {
            userPanel.setVisible(true);
            controller.getFrame().pack();
        }
    }

    public void onStopped() {
        startStopButton.setText("Start");
        startStopButton.setEnabled(true);
        selectSessionButton.setEnabled(true);
        requestServerCheckBox.setEnabled(true);
        userPanel.setVisible(false);
        controller.getFrame().pack();
    }
}

