//
//  SplashScreen.swift
//  Bulling
//
//  Loading screen with bull head logo (dartboard eyes, bowling pin horns) for macOS
//

import SwiftUI

struct SplashScreen: View {
    @State private var animateBull = false
    @State private var animateText = false

    var body: some View {
        ZStack {
            // Background gradient - bowling alley colors
            LinearGradient(
                gradient: Gradient(colors: [
                    Color(red: 0.15, green: 0.2, blue: 0.3),
                    Color(red: 0.25, green: 0.3, blue: 0.4)
                ]),
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            VStack(spacing: 30) {
                // Bull head logo
                BullHeadLogo()
                    .frame(width: 200, height: 200)
                    .scaleEffect(animateBull ? 1.0 : 0.5)
                    .opacity(animateBull ? 1.0 : 0.0)

                // App name
                Text("BULLING")
                    .font(.system(size: 56, weight: .bold, design: .rounded))
                    .foregroundColor(.white)
                    .shadow(color: Color.orange.opacity(0.5), radius: 10)
                    .opacity(animateText ? 1.0 : 0.0)
                    .offset(y: animateText ? 0 : 20)

                // Subtitle
                Text("Strike & Score!")
                    .font(.system(size: 20, weight: .medium))
                    .foregroundColor(Color.white.opacity(0.9))
                    .opacity(animateText ? 1.0 : 0.0)

                // Loading indicator
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    .scaleEffect(1.5)
                    .padding(.top, 30)
            }
        }
        .frame(minWidth: 800, minHeight: 600)
        .onAppear {
            withAnimation(.spring(response: 0.8, dampingFraction: 0.6)) {
                animateBull = true
            }
            withAnimation(.easeOut(duration: 0.6).delay(0.3)) {
                animateText = true
            }
        }
    }
}

// Bull Head Logo with dartboard eyes and bowling pin horns
struct BullHeadLogo: View {
    @State private var pulseEyes = false

    var body: some View {
        ZStack {
            // Bull head (main circle)
            Circle()
                .fill(
                    LinearGradient(
                        gradient: Gradient(colors: [
                            Color(red: 0.6, green: 0.4, blue: 0.2),
                            Color(red: 0.5, green: 0.3, blue: 0.15)
                        ]),
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .frame(width: 140, height: 140)
                .shadow(color: Color.black.opacity(0.3), radius: 10, x: 0, y: 5)

            // Left horn (bowling pin)
            BowlingPinShape()
                .fill(
                    LinearGradient(
                        gradient: Gradient(colors: [
                            Color.white,
                            Color(red: 0.95, green: 0.95, blue: 0.95)
                        ]),
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .frame(width: 25, height: 60)
                .rotationEffect(.degrees(-35))
                .offset(x: -60, y: -50)
                .shadow(color: Color.black.opacity(0.2), radius: 3)

            // Right horn (bowling pin)
            BowlingPinShape()
                .fill(
                    LinearGradient(
                        gradient: Gradient(colors: [
                            Color.white,
                            Color(red: 0.95, green: 0.95, blue: 0.95)
                        ]),
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .frame(width: 25, height: 60)
                .rotationEffect(.degrees(35))
                .offset(x: 60, y: -50)
                .shadow(color: Color.black.opacity(0.2), radius: 3)

            // Left eye (dartboard)
            DartboardEye()
                .frame(width: 35, height: 35)
                .offset(x: -25, y: -5)
                .scaleEffect(pulseEyes ? 1.1 : 1.0)

            // Right eye (dartboard)
            DartboardEye()
                .frame(width: 35, height: 35)
                .offset(x: 25, y: -5)
                .scaleEffect(pulseEyes ? 1.1 : 1.0)

            // Snout/nose
            Ellipse()
                .fill(Color(red: 0.5, green: 0.3, blue: 0.15))
                .frame(width: 50, height: 35)
                .offset(y: 25)

            // Nostrils
            Circle()
                .fill(Color.black)
                .frame(width: 8, height: 8)
                .offset(x: -10, y: 25)

            Circle()
                .fill(Color.black)
                .frame(width: 8, height: 8)
                .offset(x: 10, y: 25)

            // Ears
            Ellipse()
                .fill(Color(red: 0.6, green: 0.4, blue: 0.2))
                .frame(width: 30, height: 40)
                .rotationEffect(.degrees(-20))
                .offset(x: -70, y: -10)

            Ellipse()
                .fill(Color(red: 0.6, green: 0.4, blue: 0.2))
                .frame(width: 30, height: 40)
                .rotationEffect(.degrees(20))
                .offset(x: 70, y: -10)
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 1.0).repeatForever(autoreverses: true)) {
                pulseEyes = true
            }
        }
    }
}

// Dartboard eye design
struct DartboardEye: View {
    var body: some View {
        ZStack {
            // Outer ring (black)
            Circle()
                .fill(Color.black)

            // White ring
            Circle()
                .fill(Color.white)
                .frame(width: 28, height: 28)

            // Green ring
            Circle()
                .fill(Color.green)
                .frame(width: 20, height: 20)

            // Red ring
            Circle()
                .fill(Color.red)
                .frame(width: 12, height: 12)

            // Bullseye
            Circle()
                .fill(Color.black)
                .frame(width: 6, height: 6)
        }
    }
}

// Bowling pin shape for horns and game pins
struct BowlingPinShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()

        let width = rect.width
        let height = rect.height

        // Start at top
        path.move(to: CGPoint(x: width * 0.3, y: 0))

        // Top curve (narrow)
        path.addQuadCurve(
            to: CGPoint(x: width * 0.7, y: 0),
            control: CGPoint(x: width * 0.5, y: -2)
        )

        // Right side - bulge out
        path.addQuadCurve(
            to: CGPoint(x: width * 0.85, y: height * 0.3),
            control: CGPoint(x: width * 0.9, y: height * 0.15)
        )

        // Narrow in middle
        path.addQuadCurve(
            to: CGPoint(x: width * 0.7, y: height * 0.6),
            control: CGPoint(x: width * 0.75, y: height * 0.45)
        )

        // Widen at bottom
        path.addQuadCurve(
            to: CGPoint(x: width, y: height),
            control: CGPoint(x: width * 0.9, y: height * 0.8)
        )

        // Bottom
        path.addLine(to: CGPoint(x: 0, y: height))

        // Left side - widen at bottom
        path.addQuadCurve(
            to: CGPoint(x: width * 0.3, y: height * 0.6),
            control: CGPoint(x: width * 0.1, y: height * 0.8)
        )

        // Left side - narrow in middle
        path.addQuadCurve(
            to: CGPoint(x: width * 0.15, y: height * 0.3),
            control: CGPoint(x: width * 0.25, y: height * 0.45)
        )

        // Left side - back to top
        path.addQuadCurve(
            to: CGPoint(x: width * 0.3, y: 0),
            control: CGPoint(x: width * 0.1, y: height * 0.15)
        )

        return path
    }
}

struct SplashScreen_Previews: PreviewProvider {
    static var previews: some View {
        SplashScreen()
    }
}
