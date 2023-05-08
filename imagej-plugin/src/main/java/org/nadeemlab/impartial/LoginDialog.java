package org.nadeemlab.impartial;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

public class LoginDialog extends JDialog {

    private final JTextField usernameField;
    private final JPasswordField passwordField;
    private boolean isLoginSuccessful;

    public LoginDialog(ImpartialController controller) {
        super(controller.getFrame(), "Login", true);

        JPanel panel = new JPanel(new GridBagLayout());
        GridBagConstraints cs = new GridBagConstraints();

        cs.fill = GridBagConstraints.HORIZONTAL;

        JLabel usernameLabel = new JLabel("username");
        cs.gridx = 0;
        cs.gridy = 0;
        cs.gridwidth = 1;
        panel.add(usernameLabel, cs);

        usernameField = new JTextField(20);
        cs.gridx = 1;
        cs.gridy = 0;
        cs.gridwidth = 2;
        panel.add(usernameField, cs);

        JLabel passwordLabel = new JLabel("password");
        cs.gridx = 0;
        cs.gridy = 1;
        cs.gridwidth = 1;
        panel.add(passwordLabel, cs);

        passwordField = new JPasswordField(20);
        cs.gridx = 1;
        cs.gridy = 1;
        cs.gridwidth = 2;
        panel.add(passwordField, cs);

        panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        JButton loginButton = new JButton("Login");
        loginButton.addActionListener(e -> {
            try {
                controller.login(getUsername(), getPassword());
                isLoginSuccessful = true;
                dispose();
            } catch (IOException ex) {
                isLoginSuccessful = false;
            }
        });

        JButton cancelButton = new JButton("Cancel");
        cancelButton.addActionListener(e -> dispose());

        JButton signupButton = new JButton("Signup");
        signupButton.addActionListener(e -> {
            try {
                Desktop.getDesktop().browse(new URI(String.format("%s/signup", Config.API_URL)));
            } catch (IOException | URISyntaxException ex) {
                throw new RuntimeException(ex);
            }
        });

        JPanel buttonPanel = new JPanel();
        buttonPanel.add(loginButton);
        buttonPanel.add(signupButton);
        buttonPanel.add(cancelButton);

        getContentPane().add(panel, BorderLayout.NORTH);
        getContentPane().add(buttonPanel, BorderLayout.PAGE_END);

        pack();
        setResizable(false);
        setLocationRelativeTo(controller.getFrame());
    }

    public boolean isLoginSuccessful() {
        return isLoginSuccessful;
    }

    public String getUsername() {
        return usernameField.getText();
    }

    public String getPassword() {
        return new String(passwordField.getPassword());
    }
}
